{% if sections[""] %}
{% for type, name in definitions.items() if sections[""][type] %}
### {{ name.name }}

{% for content, issues in sections[""][type].items() %}
- {{ content|trim }} ({{ issues|join(', ') }})
{% endfor %}

{% endfor %}
{% else %}
No significant changes.
{% endif %}
