"""ScienceWorld 1.2.3 text interaction; no gold paths or privileged solutions.

Official API: allenai/ScienceWorld tag 1.2.3,
commit 5d569f9e4ff43ab6242a3622f7e25f9bc367ccd1.
"""
import asyncio
from importlib.metadata import version
from uuid import uuid4

from .base import BaseInteraction
from .agent_environment_utils import extract_action, latest_assistant_text, render_environment_observation


class ScienceworldInteraction(BaseInteraction):
    def __init__(self, config):
        super().__init__(config)
        if version("scienceworld") != "1.2.3":
            raise RuntimeError("ScienceWorld adapter requires scienceworld==1.2.3; existing ML packages must stay unchanged")
        self.states = {}
        self.observation_style = config.get('observation_style', 'compact')

    def _start(self, instance_id, task_name, variation_idx, max_steps, seed, observation_style):
        from scienceworld import ScienceWorldEnv
        # Upstream reset consumes a look-around move. We enforce assistant-action
        # limits ourselves; the Java simulator's large limit avoids ambiguous timeout done.
        env = ScienceWorldEnv(envStepLimit=1000000)
        try:
            env.load(task_name, int(variation_idx), "", generateGoldPath=False)
            obs, info = env.reset()
            self.states[instance_id] = dict(env=env, task_name=task_name, variation_idx=int(variation_idx),
                observation=obs, score=float(info["score"]), reward=0.0, done=False, steps=0,
                max_steps=int(max_steps), seed=int(seed), valid_actions=list(info["valid"]),
                action_templates=list(env.get_possible_actions()), observation_style=observation_style,
                description=info["taskDesc"])
        except BaseException:
            env.close()
            raise

    async def start_interaction(self, instance_id=None, task_name=None, variation_idx=0,
                                max_steps=100, seed=1234, observation_style=None, **kwargs):
        if not task_name:
            raise ValueError("ScienceWorld task_name required")
        instance_id = instance_id or uuid4().hex
        style = observation_style or self.observation_style
        if style not in {'compact', 'full_valid_actions'}:
            raise ValueError('Unknown ScienceWorld observation style')
        await asyncio.to_thread(self._start, instance_id, task_name, variation_idx, max_steps, seed, style)
        return instance_id

    def _render(self, state, initial=False):
        if state['observation_style'] == 'compact':
            prefix = (state['description'] + '\n\nAction templates (OBJ is an observed object):\n'
                      + '\n'.join(state['action_templates']) + '\n\n') if initial else ''
            return ('[ScienceWorld observation]\n' + prefix + state['observation']
                    + '\n\nReply with exactly one line: ACTION: <one action>')
        return render_environment_observation(benchmark="ScienceWorld",
            observation=state["description"] + "\n" + state["observation"],
            admissible_actions=state["valid_actions"])

    async def get_initial_observation(self, instance_id, **kwargs):
        return self._render(self.states[instance_id], initial=True)

    def _step(self, instance_id, action):
        state = self.states[instance_id]
        obs, reward, done, info = state["env"].step(action)
        state.update(observation=obs, score=float(info["score"]), reward=float(reward),
                     done=bool(done), valid_actions=list(info["valid"]), steps=state["steps"] + 1)
        timeout = state["steps"] >= state["max_steps"] and not done
        metrics = {k: state[k] for k in ("task_name", "variation_idx", "observation", "reward", "score", "valid_actions")}
        metrics.update(action=action, done=bool(done or timeout), step_reward=float(reward),
                       observation_style=state['observation_style'],
                       done_reason="terminal" if done else ("max_steps" if timeout else ""), seed=state["seed"])
        return bool(done or timeout), self._render(state), float(reward), metrics

    async def generate_response(self, instance_id, messages, **kwargs):
        return await asyncio.to_thread(self._step, instance_id, extract_action(latest_assistant_text(messages)))

    async def calculate_score(self, instance_id, **kwargs):
        return self.states[instance_id]["score"] / 100.0

    async def get_reward_details(self, instance_id, **kwargs):
        score = self.states[instance_id]["score"]
        return {"official_reward": score / 100.0, "training_reward": score / 100.0,
                "success": float(score >= 100), "score": score}

    async def finalize_interaction(self, instance_id, **kwargs):
        state = self.states.pop(instance_id, None)
        if state:
            await asyncio.to_thread(state["env"].close)
