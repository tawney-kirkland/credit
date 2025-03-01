SELECT
    *
FROM
    {{ ref('dim_credit_cleansed')}}
WHERE count_dependents > 20
LIMIT 10