from __future__ import annotations

import httpx

from app.config import Settings, get_settings


def _messages_to_openai(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    settings: Settings | None = None,
) -> str:
    s = settings or get_settings()
    if s.provider == "openai":
        return _openai_chat(messages, temperature, max_tokens, s)
    return _vertex_chat(messages, temperature, max_tokens, s)


def _openai_chat(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    s: Settings,
) -> str:
    from openai import OpenAI

    timeout = httpx.Timeout(s.llm_timeout_seconds)
    client = OpenAI(api_key=s.openai_api_key or None, timeout=timeout)
    resp = client.chat.completions.create(
        model=s.model,
        messages=_messages_to_openai(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()


def _vertex_chat(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    s: Settings,
) -> str:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    vertexai.init(project=s.vertex_project_id, location=s.vertex_location)
    model = GenerativeModel(s.vertex_model)
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"System:\n{content}\n")
        elif role == "user":
            parts.append(f"User:\n{content}\n")
        else:
            parts.append(f"Assistant:\n{content}\n")
    prompt = "\n".join(parts)
    cfg = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    # SDK may not expose the same timeout as OpenAI; keep generation bounded by max_tokens.
    resp = model.generate_content(prompt, generation_config=cfg)
    if not resp.candidates:
        return ""
    try:
        return (resp.text or "").strip()
    except ValueError:
        text_parts: list[str] = []
        for c in resp.candidates:
            for part in c.content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
        return "".join(text_parts).strip()
