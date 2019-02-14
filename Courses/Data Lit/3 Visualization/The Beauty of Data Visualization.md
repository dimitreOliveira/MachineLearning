### Types of Visualizations

---
In this blog, I mainly cited 
[Berkely Univ. Libbary](http://guides.lib.berkeley.edu/data-visualization/type)
[Tableau Data visualization beginner's guide](https://www.tableau.com/learn/articles/data-visualization)
[Ventsislav Yordanov's posting](https://towardsdatascience.com/data-science-with-python-intro-to-data-visualization-and-matplotlib-5f799b7c6d82)
---

## Introduction

**Data visualization** is the **graphical** representation of information and data. It is essential to **analyze** massive amounts of **information** and make data-driven **decisions**.

By using visual elements like charts, graphs, and maps, data visualization provides an accessible way to see and understand **trends, outliers, and patterns** in data. If you understand data well, you’ll have a better chance to find some insights and then be able to share your findings with other people. Some of the most famous visualizations are **line plot, scatter plot, histogram, box plot, bar chart, and pie chart**.

However, it’s not simply as easy as just dressing up a graph to make it look better. Effective data visualization is a **delicate balancing act between form and function.** The plainest graph could be too boring to catch any notice or it make tell a powerful point; the most stunning visualization could utterly fail at conveying the right message or it could speak volumes.

The data and the visuals need to work together, and there’s an art to combining great analysis with great storytelling.
![](https://www.theschool.ai/wp-content/uploads/2019/02/a1-280x300.png)
![](https://www.theschool.ai/wp-content/uploads/2019/02/a2-600x442.png)
 
[Anscombe’s Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) as an example – the quartet consists of four datasets that have nearly identical statistical properties; however, **when graphed in scatter plots,** they **reveal** four distinct **patterns.**

### **The Four Pillars of  Visualization** (Noah Illinsky at IBM)
1. Why : Has **clear purpose**
2. What : Includes **only the relevant** content
3. How : Uses **appropriate structure**
4. Everything else : Has useful **formatting**
Read more about [The Four Pillars of Visualization.](https://dl.dropboxusercontent.com/u/8059940/iliinsky%20visualization%201%20-%20pillars.pdf)

### Common general types of data visualization
* Charts
* Tables
* Graphs
* Maps
* Infographics
* Dashboards

### More specific examples of methods to visualize data
* Area Chart
* Bar Chart
* Box-and-whisker Plots
* Bubble Cloud
* Bullet Graph
* Cartogram
* Circle View
* Dot Distribution Map
* Gantt Chart
* Heat Map
* Highlight Table
* Histogram
* Matrix
* Network
* Polar Area
* Radial Tree
* Scatter Plot (2D or 3D)
* Streamgraph
* Text Tables
* Timeline
* Treemap
* Wedge Stack Graph
* Word Cloud
* And any mix-and-match combination

## How it works

### Step 1 : Think before you start
1. Think about your variables (string/categorical and numeric), the **volume of data.**
2. Think about **the question you are attempting to answer** through the visualization.
3. Think about **who will be viewing** the data and how you can best optimize the data narrative through **design.**

#### Visual characteristics
![](https://www.theschool.ai/wp-content/uploads/2019/02/perception-600x182.png)
###### Cleveland, William S., and Robert McGill. 1985, 
###### “Graphical perception and graphical methods for analyzing scientific data.” Science 299 (4716):828-833. 

### Step 2 : Remember basic visualization rules
1. Choose the appropriate plot type.
2. Label your axis when you choose plot.
3. Add a title to make our plot more informative.
4. Add labels for different categories when needed.
5. Optionally add a text or an arrow at interesting data points.
6. Use some sizes and colors of the data to make the plot more informative.

A visualization consisting of differently sized and colored **bubbles is more difficult for the human eye** to discern than a bar chart (position along a common scale).

### Step 3 : Choose type of visualization

#### 1. Bar chart
![](https://www.theschool.ai/wp-content/uploads/2019/02/barchart.jpg)

##### Detail
* Intuitive to read.
* The simplest type : one string and one numeric variable.
* Easiest for human eye perception : it uses alignment and length.
* Good for showing exact values.

##### Things to consider
* Became difficult to read when over-labeled or incorrectly labeled.
* Horizontal or vertical bars.
* Pay attention to the numerical axis of the chart  : best to start at zero.
* Order of bars : alphabetical, numerical, etc.

#### 2. Pie chart
![](https://www.theschool.ai/wp-content/uploads/2019/02/pie-600x371.png)

###### Detail
* When the **total** amount is one of your variables and you’d like to show the **subdivision** of variables.
* Best used with **one string** and **one numeric variable.**
* Show a **part-to-whole relationship.**

###### Things to consider
* The **more variables** you have, the more **difficult** the pie chart becomes to **read.**
* **Area is difficult for the eye to read.**
* If wedges are **similarly sized, try picking a different** visualization.
* **Avoid 3D** versions : notorious for causing distortion.
* **Use 2D** : easier to read while visually less stimulating.

#### 3. Line chart
![](https://www.theschool.ai/wp-content/uploads/2019/02/line-600x348.jpg)

###### Detail
* An excellent way to show **change over time.**
* Use when there are **one date** variable and **one numeric** variable.
* Can show the **continuity** : better than bar charts.

###### Things to consider
* **Difficult** to read when there are **too many lines**
* **Avoid** giving **each line its own color** : the viewer has to scan back and forth from the key to the graph
* It might be **difficult** to see where the your **data points** are.
* It’s best to **start with “zero” on the y-axis** to avoid distorting the data

#### 4. Scatter chart
![](https://www.theschool.ai/wp-content/uploads/2019/02/a2-600x442.png)

###### Detail
* **Precise** and data **dense** visualizations.
* **Correlations,** and **clusters** between **two numeric** variables.

###### Things to consider
* Not commonly used : more **difficult** for most people to read.
* **Large datasets** do not work well because **dots cover each other up.**

#### 5. Bubble chart
![](https://www.theschool.ai/wp-content/uploads/2019/02/bubble-600x254.png)

###### Detail
* A variation to the scatter plot.
* Each dot is a different size, representing an additional variable.

###### Things to consider
* Area of a circle is **difficult for the eye** to interpret.

#### Step 4 : Consider design
1. Color
![](https://www.theschool.ai/wp-content/uploads/2019/02/color_schemes-600x239.png)

* Use color meaningfully : **only use color when needed** to communicate something about the data.
* Choose the right color scheme for your data : categorical, diverging, sequential
* For categorical data, avoid using too many different colors : **no more than 6 colors is best; 12 colors max.**
* For sequential data, **don’t use rainbows, use white to highly saturated.**
* Consider the format of your visualization : displayed on a projector, in print, copied in grey scale, etc.
* Be mindful of the potential color-deficiencies of your audience
* There are tools to help choose or test color schemes that are accessible for color deficient vision.
* You may also want to **consider the cultural connotations of particular colors.**

###### 2. Scale
![](https://www.theschool.ai/wp-content/uploads/2019/02/Data_vis_example63-600x359.png)

* Use **consistent scale divisions** when graphing data that involve continuous series.
* If your data are grouped into specific spans of time, the spans should be equal.
* The histogram on the left has unequal divisions, while the histogram on the right has equal divisions.

###### 3. Axes
![](https://www.theschool.ai/wp-content/uploads/2019/02/Data_vis_example16rev-600x479.jpg)

* Vertical axes should generally **begin at the origin (zero)** not to give a misleading picture of the meaning of the data.
* The steep decline shown on this graph actually represents about a 5% change.
* It looks much greater because the vertical axis starts at a value of 2000.

###### 4. Label
* **Use sans-serif fonts.**
* Avoid all caps.
* Make sure font size large enough to be read in intended format : print, screen, etc.
* Use clear language and **avoid acronyms** in your title, legend, and labels.
* If you have only one data category, there is no need for a legend.

#### 5. Shape and Size
* Think about the aspect ratio and what is most appropriate for your data, not just what fits on the page.
* “[Banking to 45 degrees](https://eagereyes.org/basics/banking-45-degrees)” – a theory that line charts may be **more readable if their average slope is 45 degrees.**
* It is likely still a good idea to aim for 45 degrees, unless there is good reason not to.

#### 6. Normalization
![](https://www.theschool.ai/wp-content/uploads/2019/02/Data_vis_example109-600x401.jpg)

* When comparing values, differences are **not simply an artifact** of different sample or population sizes.
* This bar chart is **showing normalized and non-normalized data.**
* The **blue** bars show **total spending** on K-12 education, while the **red** bars show the same data as **spending per student.**
* It tells a different story with the population effect removed.

7. Stacked area
![](https://www.theschool.ai/wp-content/uploads/2019/02/Data_vis_example27box-281x300.jpg)

* **Difficult to interpret.**
* **Avoid** unless they illustrate clear and easily visible trends.
* In the yellow box,  the purple area declined, but it looks as though it has increased.

### Summary
* Data visualization is essential to analyze massive amounts of information and make data-driven decisions.
* The data and the visuals need to work together, and there’s an art to combining great analysis with great storytelling.
* There are 4 steps to do data visualization.
    * Step 1 : Think before you start
    * Step 2 : Remember basic visualization rules
    * Step 3 : Choose type of visualization
    * Step 4 : Consider design
* The data and the visuals need to work together, and there’s an art to combining great analysis with great storytelling.

### Further reading
* [the-single-idea-that-will-instantly-improve-your-data-visualizations](https://medium.com/swlh/the-single-idea-that-will-instantly-improve-your-data-visualizations-66f60067108a)
* [information-visualization](https://www.targetprocess.com/articles/information-visualization/)
* [https://datavizcatalogue.com/](https://datavizcatalogue.com/)
* [data-visualization-methods-in-python](https://machinelearningmastery.com/data-visualization-methods-in-python)
* [A Tour through the Visualization Zoo](http://homes.cs.washington.edu/~jheer/files/zoo/)
* [Visualization Types](http://guides.library.duke.edu/datavis/vis_types)
* [Data Visualisation Catalogue](http://www.datavizcatalogue.com/)
* [7-most-common-data-visualization-mistakes](https://thenextweb.com/dd/2015/05/15/7-most-common-data-visualization-mistakes)
* [how-to-avoid-dumb-data-visualization-mistakes](https://www.hpe.com/us/en/insights/articles/how-to-avoid-dumb-data-visualization-mistakes-1711.html)
* [Chart Dos and Don’ts](http://www.eea.europa.eu/data-and-maps/daviz/learn-more/chart-dos-and-donts)
* [More Dos and Don’ts](http://guides.library.duke.edu/datavis/topten)