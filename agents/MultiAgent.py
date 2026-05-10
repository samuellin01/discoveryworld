"""
Minimal multi-agent implementation for DiscoveryWorld.
Runs N agents in parallel (concurrent LLM calls) on the same scenario.
Each agent gets its own role/focus and shares observations via a shared knowledge store.
"""

import os
import json
import time
import copy
import signal
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from discoveryworld.ScenarioMaker import SCENARIO_NAMES, SCENARIO_INFOS, SCENARIO_DIFFICULTY_OPTIONS, getInternalScenarioName

CLAUDE_MODEL = "us.anthropic.claude-opus-4-6-v1"

def signal_handler(signum, frame):
    print("Signal handler called with signal", signum)
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class SharedKnowledge:
    """Shared knowledge store that all agents can read/write."""
    def __init__(self):
        self.entries = []

    def add(self, agent_id, entry):
        self.entries.append({"agent": agent_id, "step": len(self.entries), "content": entry})

    def get_recent(self, n=20):
        return self.entries[-n:]

    def to_string(self):
        if not self.entries:
            return "No shared knowledge yet."
        lines = []
        for e in self.entries[-20:]:
            lines.append(f"[Agent {e['agent']}] {e['content']}")
        return "\n".join(lines)


def build_agent_prompt(api, agent_idx, num_agents, agent_role, shared_knowledge, last_actions, observation):
    """Build the prompt for a single agent."""
    obs_no_vision = copy.deepcopy(observation)
    obs_no_vision.pop("vision", None)

    in_dialog = api.isAgentInDialog(agentIdx=agent_idx)

    if in_dialog:
        dialog = observation["ui"]["dialog_box"]
        prompt = f"""You are Agent {agent_idx} in a team of {num_agents} agents.
Your role: {agent_role}

*** YOU ARE CURRENTLY IN A DIALOG. You MUST choose a dialog option. ***

Dialog:
```json
{json.dumps(dialog, indent=2)}
```

Task: {json.dumps(observation['ui']['taskProgress'], indent=2)}

Shared knowledge from all agents:
{shared_knowledge.to_string()}

You MUST respond with ONLY this JSON format:
{{
  "chosen_dialog_option_int": <integer - the number of the dialog option to select>,
  "explanation": "why you chose this option",
  "shared_knowledge_update": "any useful information learned from the dialog to share with other agents"
}}

IMPORTANT: The response must contain "chosen_dialog_option_int" as an integer. Do NOT include "action", "arg1", or "arg2" fields.
"""
        return prompt

    prompt = f"""You are Agent {agent_idx} in a team of {num_agents} agents playing a scientific discovery game together.
Your role: {agent_role}

You are all working on the SAME task in the SAME environment simultaneously. Coordinate by reading shared knowledge.

Task: {json.dumps(observation['ui']['taskProgress'], indent=2)}

Your current observation:
```json
{json.dumps(obs_no_vision, indent=2)}
```

Available actions:
```json
{json.dumps(api.listKnownActions(limited=False), indent=2)}
```

Teleport locations:
```json
{json.dumps(api.listTeleportLocationsDict(), indent=2)}
```

Your last action:
```json
{json.dumps(last_actions.get(agent_idx, "No previous action"), indent=2)}
```

Shared knowledge from all agents:
{shared_knowledge.to_string()}

IMPORTANT RULES:
- Actions are JSON: {{"action": "USE", "arg1": UUID1, "arg2": UUID2}}
- For MOVE_DIRECTION: arg1 is "north"/"south"/"east"/"west"
- For TELEPORT_TO_LOCATION: arg1 is a location name
- For TELEPORT_TO_OBJECT: arg1 is an object UUID
- TELEPORT_TO_OBJECT is the fastest way to move — use it whenever possible.
- To TALK to an NPC, you must be adjacent and facing them. arg1 is their UUID.
- If your last action failed, try something DIFFERENT. Do not repeat failed actions.
- If the other agent is already doing something, do something else — don't duplicate work.

Respond with JSON only:
{{
  "action": "...",
  "arg1": ...,
  "arg2": ...,
  "explanation": "what you're doing and why",
  "shared_knowledge_update": "any findings to share with other agents (measurements, discoveries, what you've tried)"
}}
"""
    return prompt


def extract_json(text):
    """Extract JSON from response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Try code block
    start = text.find("```json")
    if start != -1:
        start = text.index("\n", start) + 1
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end])
            except:
                pass

    start = text.find("```")
    if start != -1:
        start = text.index("\n", start) + 1
        end = text.find("```", start)
        if end != -1:
            try:
                return json.loads(text[start:end])
            except:
                pass

    # Try finding first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass

    return None


def call_llm(client, prompt, max_retries=3):
    """Call Claude via Bedrock."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception as e:
            print(f"  LLM error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def run_multi_agent(scenario_name, difficulty, seed=0, num_agents=2, num_steps=100):
    """Run multiple agents on the same scenario."""
    print(f"=== Multi-Agent DiscoveryWorld ===")
    print(f"Scenario: {scenario_name} | Difficulty: {difficulty} | Agents: {num_agents} | Steps: {num_steps}")

    # Init API with multiple agents
    api = DiscoveryWorldAPI(threadID=9999)
    success = api.loadScenario(
        scenarioName=scenario_name,
        difficultyStr=difficulty,
        randomSeed=seed,
        numUserAgents=num_agents,
    )
    if not success:
        print("Failed to load scenario.")
        return

    # The API may report more "user agents" than requested due to a bug where
    # some NPCs (e.g. colonists) don't set isNPC=True. Cap to what we asked for.
    actual_agents = min(num_agents, api.numUserAgents)
    print(f"Loaded scenario. API reports {api.numUserAgents} user agents, using {actual_agents}.")
    api.numUserAgents = actual_agents

    # Create Bedrock client
    client = anthropic.AnthropicBedrock(aws_region="us-east-1")

    # Define agent roles
    roles = [
        "Explorer & Data Collector: Focus on finding objects, picking up instruments, and collecting measurements. Share all measurements with the team.",
        "Analyst & Experimenter: Focus on analyzing data, forming hypotheses, talking to NPCs, and testing theories. Use shared measurements to guide experiments.",
        "Resource Manager & Executor: Focus on managing inventory, preparing materials, and executing the final solution based on team findings.",
        "Scout & Communicator: Focus on exploring new areas, talking to NPCs, reading documents, and sharing information.",
        "Specialist: Focus on whatever subtask needs the most help based on shared knowledge.",
    ]

    shared_knowledge = SharedKnowledge()
    last_actions = {}
    progress_file = f"progress_multiagent_{scenario_name.replace(' ', '_')}_{num_agents}agents.txt"

    # Clear progress file
    with open(progress_file, "w") as f:
        f.write(f"=== Multi-Agent Run: {scenario_name} | {num_agents} agents ===\n\n")

    for step in range(num_steps):
        print(f"\n{'='*60}")
        print(f"Step {step}/{num_steps}")
        print(f"{'='*60}")

        # Get observations for all agents
        observations = {}
        for agent_idx in range(api.numUserAgents):
            observations[agent_idx] = api.getAgentObservation(agentIdx=agent_idx)

        # Build prompts for all agents
        prompts = {}
        for agent_idx in range(api.numUserAgents):
            role = roles[agent_idx % len(roles)]
            prompts[agent_idx] = build_agent_prompt(
                api, agent_idx, api.numUserAgents, role,
                shared_knowledge, last_actions, observations[agent_idx]
            )

        # Call LLM for all agents in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=api.numUserAgents) as executor:
            futures = {
                executor.submit(call_llm, client, prompts[agent_idx]): agent_idx
                for agent_idx in range(api.numUserAgents)
            }
            for future in as_completed(futures):
                agent_idx = futures[future]
                try:
                    results[agent_idx] = future.result()
                except Exception as e:
                    print(f"  Agent {agent_idx} failed: {e}")
                    results[agent_idx] = None

        # Parse responses and execute actions
        step_log = f"Step {step}:\n"
        for agent_idx in range(api.numUserAgents):
            response_text = results.get(agent_idx)
            if response_text is None:
                print(f"  Agent {agent_idx}: No response, skipping.")
                step_log += f"  Agent {agent_idx}: SKIPPED (no response)\n"
                continue

            action_json = extract_json(response_text)
            if action_json is None:
                print(f"  Agent {agent_idx}: Failed to parse JSON.")
                step_log += f"  Agent {agent_idx}: PARSE ERROR\n"
                continue

            # Share knowledge
            knowledge_update = action_json.get("shared_knowledge_update", "")
            if knowledge_update:
                shared_knowledge.add(agent_idx, knowledge_update)

            # Build action command — dialog uses different format
            if "chosen_dialog_option_int" in action_json:
                action_command = {
                    "chosen_dialog_option_int": action_json["chosen_dialog_option_int"],
                }
            else:
                action_command = {
                    "action": action_json.get("action", ""),
                    "arg1": action_json.get("arg1"),
                    "arg2": action_json.get("arg2"),
                }

            try:
                result = api.performAgentAction(agentIdx=agent_idx, actionJSON=action_command)
            except Exception as e:
                print(f"  Agent {agent_idx}: Action execution error: {e}")
                # If in dialog, try to exit it
                try:
                    ui = api.ui[agent_idx]
                    if ui.currentAgent.isInDialog():
                        ui.currentAgent.exitDialog()
                        print(f"  Agent {agent_idx}: Force-exited dialog.")
                except:
                    pass
                result = {"success": False, "errors": [str(e)]}
            last_actions[agent_idx] = {
                "action": action_json.get("action"),
                "explanation": action_json.get("explanation", ""),
                "result": str(result.get("success", False)),
            }

            explanation = action_json.get("explanation", "")
            print(f"  Agent {agent_idx}: {action_json.get('action', '?')} -> {result.get('success', '?')}")
            step_log += f"  Agent {agent_idx} [{roles[agent_idx % len(roles)][:20]}]: {action_json.get('action','?')}\n"
            step_log += f"    Explanation: {explanation}\n"
            step_log += f"    Result: {result.get('success', '?')}\n"
            if knowledge_update:
                step_log += f"    Shared: {knowledge_update}\n"

        # Tick the world
        api.tick()

        # Check score
        scorecard = api.getTaskScorecard()
        score = scorecard[0]['scoreNormalized'] if scorecard else 0
        completed = scorecard[0].get('completed', False) if scorecard else False
        step_log += f"  Score: {score:.0%} | Completed: {completed}\n"
        step_log += f"  Shared Knowledge ({len(shared_knowledge.entries)} entries)\n"
        step_log += "---\n"

        # Write to progress file
        with open(progress_file, "a") as f:
            f.write(step_log)

        print(f"  Score: {score:.0%}")

        if completed:
            print(f"\n*** TASK COMPLETED at step {step}! ***")
            break

    # Final scorecard
    scorecard = api.getTaskScorecard()
    print(f"\nFinal scorecard:")
    print(json.dumps(scorecard, indent=2))

    with open(progress_file, "a") as f:
        f.write(f"\n=== FINAL SCORECARD ===\n{json.dumps(scorecard, indent=2)}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent DiscoveryWorld")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, required=True)
    parser.add_argument("--difficulty", choices=["Easy", "Normal", "Challenge"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numAgents", type=int, default=2)
    parser.add_argument("--numSteps", type=int, default=100)
    parser.add_argument("--model", default=CLAUDE_MODEL)
    args = parser.parse_args()

    CLAUDE_MODEL = args.model
    run_multi_agent(
        scenario_name=args.scenario,
        difficulty=args.difficulty,
        seed=args.seed,
        num_agents=args.numAgents,
        num_steps=args.numSteps,
    )
