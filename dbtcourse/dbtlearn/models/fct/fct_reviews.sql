-- This is how we specify it should be
-- materialized as incremental
-- and fail on schema change
{{
  config(
    materialized = 'incremental',
    on_schema_change = 'fail'
    )
}}

WITH src_reviews AS (
    SELECT * FROM {{
        ref('src_reviews')
    }}
)

SELECT
  {{ dbt_utils.generate_surrogate_key(['listing_id', 'review_date', 'reviewer_name', 'review_text']) }}
  AS review_id,
  *
FROM src_reviews
WHERE review_text IS NOT NULL
-- This is how we specfiy the incremental part to add to the table
-- Notice this can also accomodate more specific logic as well
{% if is_incremental() %}
  AND review_date > (SELECT MAX(review_date) FROM {{ this }})
{% endif %}