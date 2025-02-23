{% macro learn_variables()%}

    {% set your_name_jinga = "Tawney" %}
    {{ log("Hello, " ~ your_name_jinga, info = True) }}

    {{ log("Hello dbt user " ~ var("user_name","NO USERNAME SET!") ~ "!", info=True)}}

{% endmacro %}