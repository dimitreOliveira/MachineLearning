### Reading Assignment (Web Scraping Techniques)
 

#### Hello, and welcome to Data-Lit!

I’ve provided you with a short reading assignment. If you go at a steady pace, it should take about 45-minutes to an hour to read through all the articles.

I put together [a colab notebook demonstrating how to collect movie posters from imdb.](https://colab.research.google.com/drive/19fp9tajYLoBARJaTEmGa-d505TIUQR_q) If you haven’t heard about colab yet, it’s a free service provided by Google to facilitate reproducible code for research. When you run a notebook, you access a virtual computer which you can use for up to 12 hours at a time.

Be sure to also check out [this notebook](https://colab.research.google.com/github/decoderkurt/web_scraper_live_demo/blob/master/web_scraper.ipynb), made by Kurt, which shows how to scrape Wikipedia for common words using regular expressions.

Good luck, and perhaps more importantly, have fun!


### Reading Assignment:
 

[Web Scraping Tutorial with Python: Tips and Tricks](https://hackernoon.com/web-scraping-tutorial-with-python-tips-and-tricks-db070e70e071)
(requests, beautiful soup, 5 minutes)

[Scraping the Internets Most Popular Websites](https://towardsdatascience.com/scraping-the-internets-most-popular-websites-a4c6f0be382d)
(guidelines for scraping, 5 minutes)

[Web-Scraping, Regular Expressions, and Data Visualization-doing it all in python](https://towardsdatascience.com/web-scraping-regular-expressions-and-data-visualization-doing-it-all-in-python-37a1aade7924)
(requests, beautiful soup, regular expressions, 5 minutes)

[Automating the boring stuff with python, Chapter 11](https://automatetheboringstuff.com/chapter11/)
(requests, beautiful soup, selenium, 20 minutes)

[Making Web Crawlers Using Scrapy for Python](https://www.datacamp.com/community/tutorials/making-web-crawlers-scrapy-python)
(scrapy, 10 minutes)


### Python libraries for web-scraping:


#### Requests

http://docs.python-requests.org/en/master/

As a preparation step to parsing, we can use Requests to download HTML and other files from the internet. Note: this library uses urllib3 under the hood.


#### Beautiful Soup

https://pypi.org/project/beautifulsoup4/

This web-scraping library is used for parsing HTML.

Because of its ease of use, it is often recommended for beginners.


#### Selenium

https://selenium-python.readthedocs.io/

Unlike Requests/Beautiful Soup, Selenium opens a visible browser window when you run the code. It can be used to simulate mouse clicks and key presses, as well as select elements of the page. One of the main use cases for this library is testing a website during development.


#### Scrapy

https://doc.scrapy.org/en/latest/index.html

Capable of ‘asynchronous networking’ (parallel, so faster), Scrapy is the most powerful and also the most difficult to learn of the libraries discussed so far.


### General tips:
 

* Right click: inspect (for details about a given web page)
* Check out a website’s robots.txt file for what they allow. Example: https://www.reddit.com/robots.txt
* Often, access by API is preferred (for simplicity).


### Excellent websites for crawling:

* https://www.wikipedia.org/
* https://soundcloud.com/ (audio – music) [allows full access to crawlers]
* https://www.imdb.com (movie info)
* https://www.rottentomatoes.com/ (movie ratings)
