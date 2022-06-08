import ast
import faulthandler
import io
import json
import os
import re
import signal
import sys
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Optional
from unittest.mock import patch, mock_open

import numpy as np
from pyext import RuntimeModule


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


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("alarm went off")
    # return
    raise TimeoutException


timeout = 10000  # seconds


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def run_test(
    test: str = None,
    debug: bool = False,
    in_outs: str = None,
    log_file=os.path.join(os.path.expanduser("~/code/gpt-neox"), "log.txt"),
):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    if test is None or in_outs is None:
        return [False]

    if debug:
        print(f"start = {datetime.now().time()}")

    in_outs = json.loads(in_outs)

    if debug:
        print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")

    if in_outs.get("fn_name") is None:
        which_type = CODE_TYPE.standard_input  # Standard input
        method_name = None
    else:
        which_type = CODE_TYPE.call_based  # Call-based
        method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded json = {datetime.now().time()}")

    results = []
    sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
    if debug:
        print(f"loading test code = {datetime.now().time()}")

    if which_type == CODE_TYPE.call_based:
        sol += test
        if debug:  # or True:
            print(f"sol = {sol}")
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            if "class Solution" not in test:
                tmp = tmp_sol
            else:
                tmp = tmp_sol.Solution()
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            print(f"type 0 compilation error = {e}")
            results.append(-2)
            return results
        signal.alarm(0)

    elif which_type == CODE_TYPE.standard_input:
        # sol
        tmp_test = test.split("\n")

        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test += "def code():\n"
                new_test += i
                started = True
            elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                new_test += "\t" + i
            else:
                new_test += i
        tmp_test = new_test

        sol += tmp_test
        if debug:
            print(f"sol = {sol}")
            # print(f"{o}")
        method_name = "code"
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            print(f"type 1 compilation error = {e}")
            results.append(-2)
            return results
        signal.alarm(0)
    if debug:
        print(f"get method = {datetime.now().time()}")

    try:
        method = getattr(tmp, method_name)  # get_attr second arg must be str
    except:
        signal.alarm(0)
        e = sys.exc_info()
        print(f"unable to get function error = {e}")
        return results

    for index, inputs in enumerate(in_outs["inputs"]):
        # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except:
            pass
        try:
            if isinstance(in_outs["outputs"][index], dict):
                in_outs["outputs"][index] = [
                    {int(k): v for k, v in in_outs["outputs"][index].items()}
                ]
        except:
            pass
        try:
            if isinstance(in_outs["outputs"][index][0], dict):
                in_outs["outputs"][index] = [
                    {int(k): v for k, v in in_outs["outputs"][index][0].items()}
                ]
        except:
            pass

        if debug:
            print(
                f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}"
            )
        if which_type == CODE_TYPE.call_based:  # Call-based
            signal.alarm(timeout)
            faulthandler.enable(
                file=open(
                    log_file,
                    mode="a",
                )
            )
            try:
                # print("------------")
                # print(inputs)
                output = method(*inputs)

                # ground truth sequences are not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == in_outs["outputs"][index]
                if (
                    isinstance(in_outs["outputs"][index], list)
                    and in_outs["outputs"][index]
                ):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                # ground truth sequences are not tuples
                try:
                    if isinstance(output[0], tuple):
                        tmp_result = tmp_result or (
                            [list(x) for x in output] == in_outs["outputs"][index][0]
                        )
                except:
                    pass
                results.append(tmp_result)

                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                faulthandler.disable()
                print(
                    f"Standard input runtime error or time limit exceeded error = {e}"
                )
                results.append(-1)
                continue
            faulthandler.disable()
            signal.alarm(0)
            if debug:
                print(
                    f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                )
        elif which_type == CODE_TYPE.standard_input:  # Standard input
            faulthandler.enable(
                file=open(
                    log_file,
                    mode="a",
                )
            )
            signal.alarm(timeout)
            passed = False

            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(in_outs["outputs"][index], list):
                in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

            with Capturing() as output:
                try:
                    call_method(method, inputs)
                    # reset the alarm
                    signal.alarm(0)
                    passed = True
                except Exception as e:
                    # runtime error or took too long
                    signal.alarm(0)
                    print(
                        f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}"
                    )
                    results.append(-1)
                signal.alarm(0)

            if not passed:
                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                continue

            if passed and debug:
                print(
                    f"==> output = {output}, test outputs = {in_outs['outputs'][index]}"
                )

            if custom_compare_(output, in_outs["outputs"][index]):
                tmp_result = True
                results.append(tmp_result)
                continue

            # ground truth sequences are expressed as lists not tuples
            if isinstance(output, tuple):
                output = list(output)

            tmp_result = False
            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
                    if isinstance(output[0], str):
                        tmp_result = tmp_result or (
                            [e.strip() for e in output] == in_outs["outputs"][index]
                        )
            except Exception as e:
                print(f"Failed check1 exception = {e}")
                pass

            if tmp_result:
                results.append(tmp_result)
                continue

            # try one more time without \n
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = i.split("\n")
                    in_outs["outputs"][index][tmp_index] = [
                        x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                    ]
            else:
                in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                in_outs["outputs"][index] = list(
                    map(lambda x: x.strip(), in_outs["outputs"][index])
                )

            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                print(f"Failed check2 exception = {e}")
                pass

            if tmp_result:
                results.append(tmp_result)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                output = list(filter(len, output))

            if debug:
                nl = "\n"
                if not isinstance(inputs, list):
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
                else:
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )

            if tmp_result:
                results.append(tmp_result)
                continue

            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                print(f"Failed check3 exception = {e}")
                pass

            try:
                output_float = [float(e) for e in output]
                gt_float = [float(e) for e in in_outs["outputs"][index]]
                tmp_result = tmp_result or (
                    (len(output_float) == len(gt_float))
                    and np.allclose(output_float, gt_float)
                )
            except Exception as e:
                pass
            try:
                if isinstance(output[0], list):
                    output_float = [float(e) for e in output[0]]
                    gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float))
                        and np.allclose(output_float, gt_float)
                    )
            except Exception as e:
                pass

            if tmp_result:
                results.append(tmp_result)
                continue

            # try by converting the stuff into split up list
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = set(i.split())
            else:
                in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

            try:
                tmp_result = output == in_outs["outputs"][index]
            except Exception as e:
                print(f"Failed check4 exception = {e}")
                continue

            if tmp_result:
                results.append(tmp_result)
                continue

                # try by converting the output into a split up list too
            if isinstance(output, list):
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = i.split()
                output = list(filter(len, output))
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = set(i)
            else:
                output = output.split()
                output = list(filter(len, output))
                output = set(output)

            try:
                tmp_result = set(frozenset(s) for s in output) == set(
                    frozenset(s) for s in in_outs["outputs"][index]
                )
            except Exception as e:
                print(f"Failed check5 exception = {e}")

            # if they are all numbers, round so that similar numbers are treated as identical
            try:
                tmp_result = tmp_result or (
                    set(frozenset(round(float(t), 3) for t in s) for s in output)
                    == set(
                        frozenset(round(float(t), 3) for t in s)
                        for s in in_outs["outputs"][index]
                    )
                )
            except Exception as e:
                print(f"Failed check6 exception = {e}")

            if tmp_result and debug:
                print("PASSED")

            results.append(tmp_result)

            if debug:
                nl = "\n"
                if not isinstance(inputs, list):
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
                else:
                    print(
                        f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )

    return results


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def repair_program_io(code):
    """Removing the special IO signs from the program.
    Case1:
      In [n]:
      (   ....:)
      and
      Out [n]:
    Case2:
      >>>
      ...
    Args:
      code: a string, the code snippet.
    Returns:
      repaired_code: a string, the repaired code snippet.
      code_list: a list of strings, each of which is lines of the original code snippet.
        The goal is to maintain all of the original information."""

    # reg patterns for case 1
    pattern_case1_in = re.compile("In ?\[\d+\]: ?")  # flag1
    pattern_case1_out = re.compile("Out ?\[\d+\]: ?")  # flag2
    pattern_case1_cont = re.compile("( )+\.+: ?")  # flag3

    # reg patterns for case 2
    pattern_case2_in = re.compile(">>> ?")  # flag4
    pattern_case2_cont = re.compile("\.\.\. ?")  # flag5

    patterns = [
        pattern_case1_in,
        pattern_case1_out,
        pattern_case1_cont,
        pattern_case2_in,
        pattern_case2_cont,
    ]

    lines = code.split("\n")
    lines_flags = [0 for _ in range(len(lines))]

    code_list = []  # a list of strings

    # match patterns
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # pdb.set_trace()
    # repair
    if lines_flags.count(0) == len(lines_flags):  # no need to repair
        repaired_code = code
        code_list = [code]
        bool_repaired = True

    elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or re.match(
        re.compile("(0*4+5*0*)+"), lines_flags_string
    ):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while flag == 0:
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""  # clean

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += (
                    re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                )

                # clean sub_block record
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += (
                    re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                )

            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        # avoid missing the last unit
        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    if (
        not bool_repaired
    ):  # not typical, then remove only the 0-flag lines after each Out.
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += (
                    re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                )

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += (
                    re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                )

            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list
