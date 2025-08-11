from pyke import fc_rule, contexts, knowledge_engine, fact_base

class PykeTracer:
    def __init__(self):
        self.events = []
        self.new_facts = set()
        self.rule_map = {}
        self.used = set()
        self.active_rule = None
        self.patched = False

tracer = PykeTracer()

# store original functions
_orig_fc_run = fc_rule.fc_rule.run
_orig_fc_new_fact = fc_rule.fc_rule.new_fact
_orig_bind = contexts.simple_context.bind
_orig_unbind = contexts.simple_context._unbind
_orig_add_case = fact_base.fact_list.add_case_specific_fact


def _rule_start(rule):
    name = rule.name
    text = tracer.rule_map.get(name, '')
    typ = 'Use' if name not in tracer.used else 'Reuse'
    tracer.events.append(f"{typ} {name}: {text}")
    tracer.used.add(name)
    tracer.active_rule = name

def _rule_end(rule):
    tracer.events.append(f"Finish implied with {rule.name}")
    tracer.active_rule = None


def run_patch(self):
    _rule_start(self)
    try:
        return _orig_fc_run(self)
    finally:
        _rule_end(self)


def new_fact_patch(self, fact_args, n):
    _rule_start(self)
    try:
        return _orig_fc_new_fact(self, fact_args, n)
    finally:
        _rule_end(self)


def bind_patch(self, var_name, var_context, val, val_context=None):
    new = _orig_bind(self, var_name, var_context, val, val_context)
    if tracer.active_rule and new and not var_name.startswith('_'):
        if hasattr(val, 'name'):
            value = f"${val.name}"
        elif hasattr(val, 'as_data'):
            try:
                value = val.as_data(val_context or self, True)
            except Exception:
                value = str(val)
        else:
            value = val
        tracer.events.append(f"Bind ${var_name} to '{value}'")
    return new


def unbind_patch(self, var_name):
    if tracer.active_rule and not var_name.startswith('_'):
        tracer.events.append(f"Unbind ${var_name}")
    return _orig_unbind(self, var_name)

def add_case_patch(self, args):
    known = args in self.universal_facts or args in self.case_specific_facts
    if tracer.active_rule:
        fact_str = f"{self.name}(" + ', '.join(repr(a) for a in args) + ")"
        if known:
            tracer.events.append(f"Obtain an already known or implied fact: {fact_str}")
        else:
            tracer.events.append(f"Obtain a new implied fact: {fact_str}")
            tracer.new_facts.add(fact_str)
    return _orig_add_case(self, args)


def patch_pyke(rule_map):
    tracer.events.clear()
    tracer.new_facts.clear()
    tracer.used.clear()
    tracer.rule_map = rule_map
    fc_rule.fc_rule.run = run_patch
    fc_rule.fc_rule.new_fact = new_fact_patch
    contexts.simple_context.bind = bind_patch
    contexts.simple_context._unbind = unbind_patch
    fact_base.fact_list.add_case_specific_fact = add_case_patch
    tracer.patched = True


def unpatch_pyke():
    if tracer.patched:
        fc_rule.fc_rule.run = _orig_fc_run
        fc_rule.fc_rule.new_fact = _orig_fc_new_fact
        contexts.simple_context.bind = _orig_bind
        contexts.simple_context._unbind = _orig_unbind
        fact_base.fact_list.add_case_specific_fact = _orig_add_case
        tracer.patched = False
