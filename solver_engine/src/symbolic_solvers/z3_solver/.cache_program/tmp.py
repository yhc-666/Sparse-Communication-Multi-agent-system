from z3 import *

objects_sort, (White, Orange, Yellow, Blue, Red) = EnumSort('objects', ['White', 'Orange', 'Yellow', 'Blue', 'Red'])
positions_sort = IntSort()
positions = [1, 2, 3, 4, 5]
objects = [White, Orange, Yellow, Blue, Red]
pos = Function('pos', objects_sort, positions_sort)

pre_conditions = []
pre_conditions.append(Distinct([pos(b) for b in objects]))
pre_conditions.append(pos(Yellow) < pos(White))
pre_conditions.append(pos(Red) > pos(Blue))
pre_conditions.append(pos(Yellow) > pos(Orange))
pre_conditions.append(pos(Blue) > pos(White))
o0 = Const('o0', objects_sort)
pre_conditions.append(ForAll([o0], And(1 <= pos(o0), pos(o0) <= 5)))

def is_valid(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(Not(option_constraints))
    return solver.check() == unsat

def is_unsat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == unsat

def is_sat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == sat

def is_accurate_list(option_constraints):
    return is_valid(Or(option_constraints)) and all([is_sat(c) for c in option_constraints])

def is_exception(x):
    return not x


if is_valid(pos(White) == 4): print('(A)')
if is_valid(pos(Orange) == 4): print('(B)')
if is_valid(pos(Yellow) == 4): print('(C)')
if is_valid(pos(Blue) == 4): print('(D)')
if is_valid(pos(Red) == 4): print('(E)')