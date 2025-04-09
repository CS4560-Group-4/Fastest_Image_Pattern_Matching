#-----------------------------------------------------------------
# pycparser: func_calls.py
#
# Using pycparser for printing out all the calls of some function
# in a C file.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------
import sys

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
sys.path.extend(['.', '..'])

from pycparser import c_ast, parse_file

# A visitor with some state information (the funcname it's looking for)
class FuncCallVisitor(c_ast.NodeVisitor):
    def __init__(self, funcname):
        self.funcname = funcname

    def visit_FuncCall(self, node):
        if node.name.name == self.funcname:
            print('%s called at %s' % (self.funcname, node.name.coord))
        # Visit args in case they contain more func calls.
        if node.args:
            self.visit(node.args)


def show_func_calls(filename, funcname):
    ast = parse_file(filename, cpp_path='g++', use_cpp=True, cpp_args=['-E', '-nostdinc', r'-Iutils/fake_libc_include', r'-I/home/steve/Documents/Courses/Y2Q3/ParallelConcurrent/Fastest_Image_Pattern_Matching/Diagnostics/mocking'])
    v = FuncCallVisitor(funcname)
    v.visit(ast)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        func = sys.argv[2]
    else:
        filename = 'MatchTool/main.cpp'
        func = 'GetMaxValueLoc'

    show_func_calls(filename, func)