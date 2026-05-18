"""
DSPy-based agent router for APEXA-Bench evaluation.

Provides DSPyOrchestrator, a drop-in replacement for OrchestratorAgent that
uses a dspy.Predict module to classify the input query into one of the five
specialist agents instead of the keyword-score router.

Used as `--config dspy` in benchmark/eval_harness.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import dspy
from dotenv import load_dotenv

from apexa_agents import (
    ANALYSIS_AGENT,
    APEXAAgent,
    ArgoProvider,
    CALIBRATION_AGENT,
    KNOWLEDGE_AGENT,
    MOTOR_AGENT,
    OrchestratorAgent,
    VISUALIZATION_AGENT,
)


load_dotenv()


_DOMAIN_TO_AGENT: Dict[str, APEXAAgent] = {
    "calibration":   CALIBRATION_AGENT,
    "analysis":      ANALYSIS_AGENT,
    "knowledge":     KNOWLEDGE_AGENT,
    "visualization": VISUALIZATION_AGENT,
    "motor":         MOTOR_AGENT,
}


_FEW_SHOT_EXAMPLES = """\
Query: Auto-calibrate the detector using CeO2 calibrant
Domain: calibration

Query: Run the FF-HEDM workflow on /data/scan_001
Domain: analysis

Query: What is Rietveld refinement?
Domain: knowledge

Query: Plot the integrated 1D pattern
Domain: visualization

Query: Move motor samX to position 2.5 mm
Domain: motor

Query: List the files in the current directory
Domain: analysis

Query: Fetch CIF files for CeO2 from Materials Project
Domain: knowledge

Query: Show me the caked image
Domain: visualization

Query: Validate the parameter file
Domain: calibration

Query: Get the current position of motor samY
Domain: motor
"""


class _ArgoLM(dspy.LM):
    """DSPy LM adapter that routes through ArgoProvider.

    DSPy's litellm path doesn't know about Argo Gateway; we bypass it and
    call ArgoProvider directly so DSPy benefits from the same payload
    handling (max_tokens, temperature exclusions) the rest of APEXA uses.
    """

    def __init__(self, model: str, username: str):
        super().__init__(model=model, model_type="chat",
                          temperature=0.0, max_tokens=512)
        self.argo_model = model
        self.username = username

    def __call__(self, prompt=None, messages=None, **kwargs):
        msgs = messages or [{"role": "user", "content": prompt or ""}]

        async def _run():
            provider = ArgoProvider(self.username, self.argo_model)
            try:
                resp = await provider.chat(msgs, temperature=0.0)
                return resp.content
            finally:
                await provider.close()

        # The DSPy LM contract is sync, but we're often called from inside an
        # already-running event loop (e.g., the eval harness). Detect that and
        # run our coroutine in a worker thread with its own loop instead of
        # calling asyncio.run() (which raises in this case).
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if not in_loop:
            content = asyncio.run(_run())
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                content = ex.submit(asyncio.run, _run()).result()

        return [content]


class _RouteQuery(dspy.Signature):
    """Classify a beamline operator query into the correct specialist domain.

    Allowed domains:
      - calibration: detector calibration, beam center, LSD, calibrant images
      - analysis: integration, HEDM, GSAS refinement, file/dir ops, calculations
      - knowledge: definitions, explanations, literature, fetching CIFs
      - visualization: plotting, viewing, lineouts, caked images, overlays
      - motor: EPICS motor moves, jogs, position queries, limits

    Output exactly one of the domain words above.
    """

    query: str = dspy.InputField(desc="Natural-language operator query")
    domain: str = dspy.OutputField(
        desc="One of: calibration, analysis, knowledge, visualization, motor"
    )


class _DSPyRouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(_RouteQuery)

    def forward(self, query: str):
        # Inject few-shot prefix via instructions (zero-shot Predict otherwise).
        return self.predict(query=f"{_FEW_SHOT_EXAMPLES}\nQuery: {query}\nDomain:")


class DSPyOrchestrator(OrchestratorAgent):
    """Orchestrator that uses a DSPy classifier for agent routing.

    Falls back to the keyword router if the DSPy output isn't one of the
    five known domains (model produced gibberish or hit rate-limits).
    """

    def __init__(self, execute_tool_fn, all_tools: List[Dict], context=None,
                 router_model: str = "gpt5mini"):
        super().__init__(execute_tool_fn, all_tools, context=context)
        username = os.getenv("ANL_USERNAME")
        if not username:
            raise RuntimeError("ANL_USERNAME must be set for DSPyOrchestrator")
        lm = _ArgoLM(model=router_model, username=username)
        dspy.configure(lm=lm)
        self._dspy_router = _DSPyRouterModule()

    def _route(self, query: str) -> APEXAAgent:
        try:
            out = self._dspy_router(query=query)
            domain_raw = (out.domain or "").strip().lower().split()[0]
            domain = domain_raw.rstrip(".,;:")
            if domain in _DOMAIN_TO_AGENT:
                self._last_agent = _DOMAIN_TO_AGENT[domain]
                return self._last_agent
        except Exception as exc:
            print(f"  [dspy-router] fallback to keyword router: {exc}")
        # Fallback: keyword routing
        return super()._route(query)
