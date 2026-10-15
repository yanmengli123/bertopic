"""Conservative address extraction; unrecognized countries remain missing."""
import re

import pycountry

COUNTRIES = {}
for country in pycountry.countries:
    for name in [country.name, country.alpha_3, getattr(country, 'official_name', '')]:
        if name:
            COUNTRIES[re.sub(r'[^a-z]', '', name.casefold())] = country.name
COUNTRIES.update({'peoplesrchina': 'China', 'prchina': 'China', 'england': 'United Kingdom',
                  'scotland': 'United Kingdom', 'wales': 'United Kingdom', 'uk': 'United Kingdom',
                  'southkorea': 'Korea, Republic of', 'us': 'United States'})


def address_groups(value):
    # Author lists can contain semicolons inside brackets.
    value = re.sub(r'\[[^\]]*\]', '', str(value or ''))
    countries, institutions = set(), set()
    for address in value.split(';'):
        parts = [re.sub(r'\s+', ' ', p).strip(' .') for p in address.split(',') if p.strip()]
        if not parts:
            continue
        if len(parts) >= 2:
            institutions.add(parts[0])
        tail = parts[-1]
        if re.search(r'\bUSA\.?$', tail, re.I):
            countries.add('United States')
            continue
        country = COUNTRIES.get(re.sub(r'[^a-z]', '', tail.casefold()))
        if country:
            countries.add(country)
    return sorted(countries), sorted(institutions)
