{% extends "!autosummary/class.rst" %}

{% block methods %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in methods %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
{% endblock %}

{% block attributes %}
   .. HACK -- the point here is that we don't want this to appear in the output, but the autosummary should still generate the pages.
      .. autosummary::
         :toctree:
      {% for item in attributes %}
         ~{{ name }}.{{ item }}
      {%- endfor %}
{% endblock %}