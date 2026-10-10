# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IAD Agent Loop: multi-turn tool calling with <answer> termination."""

import asyncio
import json
import re
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)

_XML_INSPECT_RE = re.compile(
    r'ابه\s*<name>\s*inspect\s*</name>\s*'
    r'<crop>\s*(\[[\d\s.,\-]+\])\s*</crop>\s*'
    r'<query>\s*(.*?)\s*</query>\s*'
    r'abwe',
    re.DOTALL | re.IGNORECASE,
)


def _extract_xml_inspect_calls(response_text: str) -> List:
    """Extract inspect tool calls from XML format response."""
    from verl.experimental.agent_loop.tool_parser import FunctionCall
    calls = []
    for m in _XML_INSPECT_RE.finditer(response_text):
        try:
            crop = json.loads(m.group(1))
            query = m.group(2).strip()
            args = json.dumps({"crop": crop, "query": query})
            calls.append(FunctionCall(name="inspect", arguments=args))
        except (json.JSONDecodeError, Exception):
            pass
    return calls


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """State container for the IAD agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        extra_info: dict[str, Any] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.extra_info = extra_info or {}

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.tool_calls: list[FunctionCall] = []
        self.routed_experts = None
        self.extra_fields: dict[str, Any] = {}
        self.tool_call_history: list[dict[str, Any]] = []


@register("airiad_tool_agent")
class AiRIADAgentLoop(AgentLoopBase):
    """IAD Agent Loop with <answer> termination and tool call tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        self.resize_input_image = self.data_config.get("resize_input_image", False)
        self.max_input_image_pixels = self.data_config.get("max_input_image_pixels", 262144)
        self.min_input_image_pixels = self.data_config.get("min_input_image_pixels", 1024)

    def _compress_input_image(self, image: Any) -> Any:
        """Resize input image to fit within pixel limits."""
        import PIL.Image
        import math
        w, h = image.size
        pixels = w * h

        if pixels > self.max_input_image_pixels:
            resize_factor = math.sqrt(self.max_input_image_pixels / pixels)
            w, h = int(w * resize_factor), int(h * resize_factor)
            image = image.resize((w, h))

        if (image.width * image.height) < self.min_input_image_pixels:
            resize_factor = math.sqrt(self.min_input_image_pixels / (image.width * image.height))
            w, h = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((w, h))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        if images and self.resize_input_image:
            images = [self._compress_input_image(img) for img in images]

        metrics = {}
        request_id = uuid4().hex
        extra_info = kwargs.get("extra_info", {})
        tools_kwargs = extra_info.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            extra_info=extra_info,
        )

        state = AgentState.PENDING

        while state != AgentState.TERMINATED:
            try:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    state = AgentState.TERMINATED
            except Exception:
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[:self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({
            "tool_rewards": agent_data.tool_rewards,
            "tool_call_history": agent_data.tool_call_history,
            "assistant_turns": agent_data.assistant_turns,
            "messages": agent_data.messages,
        })
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=None,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        )

        agent_data.messages.append({"role": "assistant", "content": response_text})

        # Try XML format first, then fall back to standard ToolParser
        agent_data.tool_calls = _extract_xml_inspect_calls(response_text)
        if not agent_data.tool_calls:
            tools = [tool.tool_schema for tool in self.tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
                agent_data.response_ids, tools
            )

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        # No tool calls → check for <answer> tag
        if ANSWER_PATTERN.search(response_text):
            return AgentState.TERMINATED

        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        try:
            with simple_timer("tool_calls", agent_data.metrics):
                responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_responses = []
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    processed_responses.append((
                        ToolResponse(text=f"Error: Tool execution failed with exception: {resp}"),
                        0.0,
                        {"name": tool_call_names[i], "success": False, "error": str(resp)}
                    ))
                else:
                    processed_responses.append(resp)
            responses = processed_responses

            for tool_response, tool_reward, tool_info in responses:
                agent_data.tool_call_history.append(tool_info)

                # Build message from tool response (align with SFT format)
                if tool_response.image or tool_response.video:
                    if not getattr(self.processor, "image_processor", None):
                        message = {"role": "tool", "content": "Error: Image processing not available"}
                    else:
                        content = []
                        tool_text = tool_response.text or ""
                        if "<IMAGE_HERE>" in tool_text:
                            before, after = tool_text.split("<IMAGE_HERE>", 1)
                            content.append({"type": "text", "text": before})
                            content.append({"type": "image"})
                            content.append({"type": "text", "text": after})
                        else:
                            content.append({"type": "text", "text": tool_text})
                            content.append({"type": "image"})

                        if tool_response.video:
                            content.append({"type": "video"})
                        message = {"role": "tool", "content": content}
                else:
                    message = {"role": "tool", "content": tool_response.text or ""}

                add_messages.append(message)

                if tool_response.image:
                    if isinstance(tool_response.image, list):
                        for img in tool_response.image:
                            if img is not None:
                                new_images_this_turn.append(img)
                    else:
                        if tool_response.image is not None:
                            new_images_this_turn.append(tool_response.image)

                if tool_response.video:
                    pass

                if tool_reward is not None:
                    agent_data.tool_rewards.append(tool_reward)

            agent_data.messages.extend(add_messages)

            if self.tool_parser_name == "gpt-oss":
                tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                images = new_images_this_turn if new_images_this_turn else None
                videos = None
                response_ids = await self.apply_chat_template(
                    add_messages,
                    images=images,
                    videos=videos,
                    remove_system_prompt=True,
                )

            if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
                return AgentState.TERMINATED

            if new_images_this_turn:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]
                for img in new_images_this_turn:
                    agent_data.image_data.append(img)

            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            agent_data.user_turns += 1
            return AgentState.GENERATING

        except Exception:
            return AgentState.TERMINATED

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        tool, instance_id = None, None
        tool_info = {"name": tool_call.name, "arguments": {}, "success": False}

        class_name = agent_data.extra_info.get("class_name", "unknown")

        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool_info["arguments"] = tool_args

            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            create_kwargs = kwargs.get("create_kwargs", {})

            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            tool_info["success"] = True
            tool_info["result"] = res if res else {}

        except Exception as e:
            tool_info["error"] = str(e)
            return (
                ToolResponse(text=f"Error when executing tool: {e}"),
                0.0,
                tool_info,
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        # Build tool response text (aligned with SFT_TOOL_RESULT_TEMPLATE)
        crop = tool_args.get("crop", [])
        query = tool_args.get("query", "")
        knowledge_text = tool_execution_response.text or ""

        _verify = (
            "---\n"
            "Now verify your hypothesis:\n"
            "- Does the cropped image tightly cover an anomaly?\n"
            "- Does the retrieved knowledge accurately describe what you see?\n\n"
            "If YES to both → output <answer>\n"
            "If NO to either → refine and call inspect again"
        )
        if knowledge_text:
            tool_response_text = (
                f"Tool result for inspect(crop={crop}, query=\"{query}\"):\n\n"
                "<IMAGE_HERE>"
                f"\n\nRetrieved knowledge for \"{class_name}\":\n"
                f"{knowledge_text}\n\n"
                + _verify
            )
        else:
            tool_response_text = (
                f"Tool result for inspect(crop={crop}, query=\"{query}\"):\n\n"
                "<IMAGE_HERE>\n\n"
                + _verify
            )

        # Truncate if needed
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        tool_response_kwargs = {"text": tool_response_text}

        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, tool_info
