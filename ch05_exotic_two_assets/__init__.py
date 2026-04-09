# Register underscore-prefixed aliases for numeric module names
import importlib, sys

_aliases = {
    '_01_margrabe_exchange': '01_margrabe_exchange',
    '_02_spread_max_min_options': '02_spread_max_min_options',
    '_03_currency_translated_fx_options': '03_currency_translated_fx_options',
    '_04_rainbow_options': '04_rainbow_options',
}
for _alias, _real in _aliases.items():
    _full_alias = f'{__name__}.{_alias}'
    if _full_alias not in sys.modules:
        try:
            sys.modules[_full_alias] = importlib.import_module(f'{__name__}.{_real}')
        except ImportError:
            pass
