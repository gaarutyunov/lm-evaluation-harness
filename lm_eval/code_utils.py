import ast
import io
from typing import Optional


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False,
        },
    )

    return ret.getvalue()


def _find_indentation(line, config):
    if len(line) and line[0] in (" ", "\t") and not line.isspace():
        if line[0] == "\t":
            config["is-tabs"] = True
        # Find indentation
        i = 0
        for char in list(line):
            if char not in (" ", "\t"):
                break
            i += 1
        config["from"] = i


def find_indentation(line, config):
    # Find indentation level used in file
    if config["from"] < 0:
        _find_indentation(line, config)

    if config["from"] >= 0:
        # Set old indent
        indent = " " if not config["is-tabs"] else "\t"
        indent = indent * config["from"]

        # Set new indent
        newindent = " " if not config["tabs"] else "\t"
        if not config["tabs"]:
            newindent = newindent * config["to"]

        return indent, newindent

    # Continue to the next line, indentation not found
    return False


def replace_inline_tabs(content, config):
    newcontent = ""
    imagined_i = 0
    for i in range(0, len(content)):
        char = content[i]
        if char == "\t":
            spaces = config["tabsize"] - (imagined_i % config["tabsize"])
            newcontent += " " * spaces
            imagined_i += spaces
        else:
            newcontent += char
            imagined_i += 1
    return newcontent


def run(fd_in, fd_out, config):
    while True:
        line = fd_in.readline()
        if not line:
            break
        line = line.rstrip("\r\n")

        # Find indentation style used in file if not set
        if config["from"] < 0:
            indent = find_indentation(line, config)
            if not indent:
                print(line, file=fd_out)
                continue
            indent, newindent = indent

        # Find current indentation level
        level = 0
        while True:
            whitespace = line[: len(indent) * (level + 1)]
            if whitespace == indent * (level + 1):
                level += 1
            else:
                break

        content = line[len(indent) * level :]
        if config["all-tabs"]:
            content = replace_inline_tabs(content, config)

        line = (newindent * level) + content
        print(line, file=fd_out)


def extract_func_signature_and_body(code: str) -> tuple[Optional[str], Optional[str]]:
    nodes = ast.parse(code)
    sig, body = None, None

    for node in ast.walk(nodes):
        if isinstance(node, ast.FunctionDef):
            sig = "def "

            sig += node.name + "("

            for arg in node.args.args:
                sig += arg.arg + ", "

            sig = sig.rstrip(", ") + ")"

            body = ast.unparse(node.body)
            break

    return sig, body
