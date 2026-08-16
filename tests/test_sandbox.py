from qwen_translation_test.sandbox import MontyPythonSandbox, SandboxSettings


def settings(**overrides: object) -> SandboxSettings:
    values = {
        "enabled": True,
        "timeout_seconds": 1.0,
        "max_memory_bytes": 16 * 1024 * 1024,
        "max_recursion_depth": 100,
        "max_code_chars": 2_000,
        "max_output_bytes": 2_000,
        "max_tool_rounds": 2,
    }
    values.update(overrides)
    return SandboxSettings(**values)


def test_monty_executes_python_and_collects_output() -> None:
    sandbox = MontyPythonSandbox(settings())

    result = sandbox.execute("values = [2, 3, 5]\nprint(sum(values))\nsum(values)")

    assert result.success is True
    assert result.result == 10
    assert result.stdout == "10\n"
    assert result.stderr == ""


def test_monty_blocks_ungranted_host_file_access() -> None:
    sandbox = MontyPythonSandbox(settings())

    result = sandbox.execute("open('/etc/passwd').read()")

    assert result.success is False
    assert result.error is not None
