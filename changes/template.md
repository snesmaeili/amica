{% if sections[""] -%}
{% for type, name in definitions.items() if sections[""][type] %}

### {{ name.name }}

{% for content, issues in sections[""][type].items() %}

- {{ content | trim }}{% if issues %} ({{ issues | join(", ") }}){% endif %}
  {% endfor %}
  {% endfor %}
  {%- else -%}
  No significant changes.
  {%- endif %}
