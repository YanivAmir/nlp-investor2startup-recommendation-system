# Recommendation System for Investors to Startups

This is a script I wrote during my internship in early 2021 after completing my ML studies.

The goal was to make bilateral recommendations of Investors to Stratups based on similar interests, markets, technologies etc.

The text vectorisation is based on TF-IDF and is pretty straight forward, with both multiple-word ccombinations, and sub-word charachters used for tokenisation.

The main challenge was to look for a uniform and consice information system that provides unbiased desciptive text for both stratup and investors.
This was solved by scraping google search results snippets, when the company name is being queried, and combining them into a company descriptive text.
Similarly for investors, a list of previous company investments is required (this could be companies that the particular investor is interested in modelling their next investment on). The model finds the descriptive texts of each the profiling investments of the investor.

Matching is either based on the best matching out of all profiling companies in the investors portfolio to the startup in question, or by averaging the entire portfolio and matching the startup that best fits the artificial "average" investment of the investor.

Result is a dataframe where each investor is given the list of best matching recommended startups for investments, in decreaseing order, and their relative scores.
Result is also given in the perspective of the startups, i.e. which of the list of investors given is the ideal candidate for investment in the startup based on common market niches, technology etc.
A sample small data is given for testing.

