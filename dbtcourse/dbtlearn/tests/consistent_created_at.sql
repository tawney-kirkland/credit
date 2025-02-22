SELECT
    *
FROM {{ ref("dim_listings_cleansed")}} AS lc
LEFT JOIN {{ ref("fct_reviews")}} as fr 
ON (lc.listing_id = fr.listing_id)
WHERE review_date < created_at