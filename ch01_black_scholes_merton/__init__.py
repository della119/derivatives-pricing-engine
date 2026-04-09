# Register underscore-prefixed aliases for numeric module names
# so that cross-module imports like
#   from ch01_black_scholes_merton._04_generalized_bsm import generalized_bsm
# resolve correctly.
import importlib, sys

_aliases = {
    '_01_black_scholes_1973': '01_black_scholes_1973',
    '_02_merton_1973': '02_merton_1973',
    '_03_black_76': '03_black_76',
    '_04_generalized_bsm': '04_generalized_bsm',
    '_05_garman_kohlhagen': '05_garman_kohlhagen',
    '_06_bachelier_sprenkle_boness_samuelson': '06_bachelier_sprenkle_boness_samuelson',
}
for _alias, _real in _aliases.items():
    _full_alias = f'{__name__}.{_alias}'
    if _full_alias not in sys.modules:
        try:
            sys.modules[_full_alias] = importlib.import_module(f'{__name__}.{_real}')
        except ImportError:
            pass
