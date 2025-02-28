WITH src_credit AS (
    SELECT * FROM {{ ref('src_credit')}}
)

SELECT
    serious_delinquency,
    credit_usage_pct,
    age,
    count_3059_days_past_due,
    debt_ratio,
    CASE WHEN
        monthly_income = 'NA' THEN NULL
        ELSE monthly_income
    END AS monthly_income,
    credit_loans_count,
    count_90_days_late,
    real_estate_line_loans_count,
    count_6089_days_past_due,
    CASE WHEN
        count_dependents = 'NA' THEN 0
        ELSE count_dependents
    END AS count_dependents
FROM src_credit