# Recommendation System for Investors to Startups

This is a script I wrote during my internship in early 2021 after completing my ML studies.

The goal was to make bilateral recommendations of Investors to Stratups based on similar interests, markets, technologies etc.

Tech used:
<img alt="Python" src="https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white&style-flat">
<img alt="scikit-learn" src="https://img.shields.io/badge/Scikit-f7931e?logo=scikit-learn&logoColor=white&style-flat">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-013242?logo=numpy&logoColor=white&style-flat">
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style-flat">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-f37626?logo=jupyter&logoColor=white&style-flat">
<img alt="PyCharm" src="https://img.shields.io/badge/PyCharm-000000?logo=PyCharm&logoColor=white&style-flat">
<img alt="Google" src="https://img.shields.io/badge/Google-4285f4?logo=Google&logoColor=white&style-flat">


The text vectorisation is based on TF-IDF and is pretty straight forward, with both multiple-word ccombinations, and sub-word charachters used for tokenisation.

The main challenge was to look for a uniform and consice information system that provides unbiased desciptive text for both stratup and investors.
This was solved by scraping google search results snippets, when the company name is being queried, and combining them into a company descriptive text.
Similarly for investors, a list of previous company investments is required (this could be companies that the particular investor is interested in modelling their next investment on). The model finds the descriptive texts of each the profiling investments of the investor.

Matching is either based on the best matching out of all profiling companies in the investors portfolio to the startup in question, or by averaging the entire portfolio and matching the startup that best fits the artificial "average" investment of the investor.

Result is a dataframe where each investor is given the list of best matching recommended startups for investments, in decreaseing order, and their relative scores.
Result is also given in the perspective of the startups, i.e. which of the list of investors given is the ideal candidate for investment in the startup based on common market niches, technology etc.
A sample small data is given for testing.

## Results from Sample Data:

Sample Data Startup Companies:

<img width="212" alt="Screenshot 2024-10-27 at 21 30 50" src="https://github.com/user-attachments/assets/9ba52ffe-c3d2-4138-adee-cf14a7616055">

Sample Data Investors:

<img width="408" alt="Screenshot 2024-10-27 at 21 31 13" src="https://github.com/user-attachments/assets/582f4e17-132d-4903-ae22-085da03b001c">

Example Company-Descriptive Text Mined from Google Search Results:
Gucci:
<img width="1255" alt="Screenshot 2024-10-27 at 21 32 03" src="https://github.com/user-attachments/assets/1ae43787-ee2f-4393-90dc-b72bf5830cee">
Tesla:
<img width="1249" alt="Screenshot 2024-10-27 at 21 32 28" src="https://github.com/user-attachments/assets/974ca6b6-52d0-4922-8c6c-5248d48fa3bf">

Distribution of Text Lengths in Small Sample Data:
<img width="621" alt="Screenshot 2024-10-27 at 21 52 28" src="https://github.com/user-attachments/assets/eb9daf7d-b5e2-4872-9dd5-f6861af081e2">

Results When Investors' Profiling Companies are Averaged:
<img width="1087" alt="image" src="https://github.com/user-attachments/assets/57b5ef2a-217c-48a8-98a9-4765487e0340">

<img width="683" alt="Screenshot 2024-10-27 at 21 56 13" src="https://github.com/user-attachments/assets/4a18f42a-35fe-4e87-959e-d9a8c23a0d21">

Results When Investor's Profiling Companies are Measured Individually and the Max Score is Taken:
<img width="1085" alt="image" src="https://github.com/user-attachments/assets/95f1b5b1-2fdf-40a5-8ac2-0dee3dd7eb4e">
The change between the two options is not so clear with this small sample.







