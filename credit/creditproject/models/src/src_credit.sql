WITH src_credit AS (
    SELECT * FROM {{ source('credit', 'client_credit')}}
)

SELECT
    SERIOUSDLQIN2YRS AS serious_delinquency,
    REVOLVINGUTILIZATIONOFUNSECUREDLINES AS credit_usage_pct,
    age,
    NUMBEROFTIME30_59DAYSPASTDUENOTWORSE AS count_3059_days_past_due,
    DEBTRATIO AS debt_ratio,
    MONTHLYINCOME AS monthly_income,
    NUMBEROFOPENCREDITLINESANDLOANS AS credit_loans_count,
    NUMBEROFTIMES90DAYSLATE AS count_90_days_late,
    NUMBERREALESTATELOANSORLINES AS real_estate_line_loans_count,
    NUMBEROFTIME60_89DAYSPASTDUENOTWORSE AS count_6089_days_past_due,
    NUMBEROFDEPENDENTS AS count_dependents,
    ROW_NUMBER() OVER(ORDER BY CURRENT_TIMESTAMP()) AS unique_id
FROM
    src_credit