# tracer.py - Enhanced version with original constraint rules
from time import perf_counter
import inspect
from constraint import (
    BacktrackingSolver, RecursiveBacktrackingSolver, Domain,
    Problem
)

###############################################################################
# 全局约束规则存储
###############################################################################
_constraint_rules = {}  # 存储约束索引到原始规则的映射
_constraint_counter = 0  # 约束计数器

def set_constraint_rule(index, rule):
    """设置约束规则"""
    global _constraint_rules
    _constraint_rules[index] = rule

def get_constraint_rule(index):
    """获取约束规则"""
    return _constraint_rules.get(index, "unknown constraint")

def clear_constraint_rules():
    """清空约束规则"""
    global _constraint_rules, _constraint_counter
    _constraint_rules = {}
    _constraint_counter = 0

###############################################################################
# 1) 事件收集器 —— 线程安全、可被多次导入复用
###############################################################################
class TraceCollector:
    def __init__(self):
        self.events = []          # 存储所有事件
        self.node_cnt = 0
        self.fail_cnt = 0
        self.max_depth = 0
        self.t0 = perf_counter()

    def log(self, typ, depth, **data):
        self.events.append((typ, depth, data))
        self.max_depth = max(self.max_depth, depth)
        if typ == "TRY":
            self.node_cnt += 1
        elif typ == "FAIL":
            self.fail_cnt += 1

    # 方便外部拿到最终 trace
    def dump(self):
        dur = perf_counter() - self.t0
        summary = {
            "nodes": self.node_cnt,
            "fails": self.fail_cnt,
            "max_depth": self.max_depth,
            "time_sec": round(dur, 6),
        }
        return {"trace": self.events, "summary": summary}

    def clear(self):
        """清空追踪记录，用于新的求解"""
        self.events = []
        self.node_cnt = 0
        self.fail_cnt = 0
        self.max_depth = 0
        self.t0 = perf_counter()


###############################################################################
# 2) 带追踪的 Solver —— 基于 BacktrackingSolver 改造，直接返回违反的规则
###############################################################################
class TracingSolver(BacktrackingSolver):
    def __init__(self, collector=None, *a, **kw):
        super().__init__(*a, **kw)
        self._collector = collector or TraceCollector()

    def _get_violated_rule(self, constraint, variables, assignments, failed_variable, failed_value):
        """
        直接返回违反的原始约束规则
        """
        # 尝试匹配约束到原始规则
        constraint_type = type(constraint).__name__
        
        # 根据约束类型和变量匹配原始规则
        if constraint_type == "AllDifferentConstraint":
            # 查找AllDifferentConstraint规则
            for rule in _constraint_rules.values():
                if "AllDifferentConstraint" in rule:
                    # 检查是否有重复值
                    current_assignment = assignments.copy()
                    current_assignment[failed_variable] = failed_value
                    values = [current_assignment.get(var) for var in variables if var in current_assignment]
                    value_counts = {}
                    for val in values:
                        if val is not None:
                            value_counts[val] = value_counts.get(val, 0) + 1
                    
                    duplicates = [val for val, count in value_counts.items() if count > 1]
                    if duplicates:
                        return f"violates: {rule} (value {duplicates[0]} appears multiple times)"
                    else:
                        return f"violates: {rule}"
        
        elif constraint_type == "FunctionConstraint":
            # 根据涉及的变量匹配函数约束
            var_set = set(variables)
            
            # 查找匹配的规则
            for rule in _constraint_rules.values():
                if "AllDifferentConstraint" not in rule:
                    # 提取规则中的变量
                    import re
                    rule_vars = set(re.findall(r'\b[a-zA-Z_]+\b', rule))
                    # 移除数字和操作符
                    rule_vars = {var for var in rule_vars if not var.isdigit() and var not in ['and', 'or', 'not']}
                    
                    # 如果变量集合匹配，这很可能是对应的规则
                    if rule_vars == var_set:
                        return f"violates: {rule}"
            
            # 如果没有找到精确匹配，尝试部分匹配
            for rule in _constraint_rules.values():
                if "AllDifferentConstraint" not in rule:
                    # 检查失败变量是否在规则中
                    if failed_variable in rule:
                        return f"violates: {rule}"
        
        # 如果都没有匹配到，返回通用信息
        return f"violates constraint involving {', '.join(variables)}"

    # ---- 重写 getSolutionIter -------------------------------------------------
    # 基本照抄原实现，但在关键位置插桩并添加失败原因分析
    def getSolutionIter(self, domains, constraints, vconstraints):
        forwardcheck = self._forwardcheck
        assignments = {}
        queue = []

        while True:
            # Mix the Degree and Minimum Remaing Values (MRV) heuristics
            lst = [(-len(vconstraints[variable]), len(domains[variable]), variable) for variable in domains]
            lst.sort(key=lambda x: (x[0], x[1]))
            for item in lst:
                if item[-1] not in assignments:
                    # Found unassigned variable
                    variable = item[-1]
                    values = domains[variable][:]
                    if forwardcheck:
                        pushdomains = [domains[x] for x in domains if x not in assignments and x != variable]
                    else:
                        pushdomains = None
                    break
            else:
                # No unassigned variables. We've got a solution. Go back
                # to last variable, if there's one.
                depth = len(assignments)
                self._collector.log("SUCCESS", depth, assignments=assignments.copy())
                yield assignments.copy()
                if not queue:
                    return
                variable, values, pushdomains = queue.pop()
                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            while True:
                # We have a variable. Do we have any values left?
                if not values:
                    # No. Go back to last variable, if there's one.
                    depth = len(assignments)
                    self._collector.log("BACKTRACK", depth, variable=variable, reason="no more values to try")
                    del assignments[variable]
                    while queue:
                        variable, values, pushdomains = queue.pop()
                        if pushdomains:
                            for domain in pushdomains:
                                domain.popState()
                        if values:
                            break
                        del assignments[variable]
                    else:
                        return

                # Got a value. Check it.
                value = values.pop()
                depth = len(assignments)
                self._collector.log("TRY", depth, variable=variable, value=value)
                assignments[variable] = value

                if pushdomains:
                    for domain in pushdomains:
                        domain.pushState()

                for constraint, variables in vconstraints[variable]:
                    if not constraint(variables, domains, assignments, pushdomains):
                        # Value is not good - 返回违反的具体规则
                        depth = len(assignments) - 1
                        violated_rule = self._get_violated_rule(constraint, variables, assignments, variable, value)
                        self._collector.log("FAIL", depth, variable=variable, value=value, reason=violated_rule)
                        break
                else:
                    break

                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            # Push state before looking for next variable.
            queue.append((variable, values, pushdomains))

    # 对外暴露 trace
    @property
    def trace(self):
        return self._collector.dump()


###############################################################################
# 3) 打补丁接口 —— enable_tracing()
###############################################################################
_collector_singleton = TraceCollector()    # 保证全局唯一

def enable_tracing():
    """
    动态替换默认求解器。执行一次即可（多次调用安全）。
    """
    # 清空之前的追踪记录和约束规则
    _collector_singleton.clear()
    clear_constraint_rules()
    
    # 把回溯求解器类都指到 TracingSolver
    import constraint
    constraint.BacktrackingSolver = TracingSolver
    constraint.RecursiveBacktrackingSolver = TracingSolver
    
    # 让 Problem() 默认选我们的 solver
    original_init = Problem.__init__

    def patched_init(self, solver=None):
        # 若用户强行传了其它 solver，不动；
        # 否则给它换成 TracingSolver
        if solver is None:
            solver = TracingSolver(_collector_singleton)
        original_init(self, solver=solver)

    Problem.__init__ = patched_init
    return _collector_singleton

def get_trace():
    """供外部拿结果用。"""
    return _collector_singleton.dump()

def trace_to_text(trace_data):
    """将追踪数据转换为可读的文本格式，包含违反的具体规则"""
    if not trace_data or "trace" not in trace_data:
        return "No trace data available"
    
    lines = []
    lines.append("=== CSP Solving Trace ===")
    #lines.append(f"Summary: {trace_data['summary']}")
    lines.append("")
    
    for event_type, depth, data in trace_data["trace"]:
        indent = "  " * depth
        if event_type == "TRY":
            lines.append(f"{indent}TRY: {data['variable']} = {data['value']}")
        elif event_type == "FAIL":
            reason = data.get('reason', 'unknown constraint violation')
            lines.append(f"{indent}FAIL: {data['variable']} = {data['value']} ({reason})")
        elif event_type == "SUCCESS":
            lines.append(f"{indent}SUCCESS: {data['assignments']}")
        elif event_type == "BACKTRACK":
            reason = data.get('reason', 'backtracking')
            lines.append(f"{indent}BACKTRACK: {data['variable']} ({reason})")
    
    return "\n".join(lines) 