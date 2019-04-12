### Dimensionality Reduction

## Dimensionality
### Reduction

Hello Wizards!!! Hope you are all doing so great. I almost always encourage myself to answer 3 important questions before starting any work.

1. What? – What is this work about? I need to understand that first.
2. Why? – Again why am I doing this? Need this to scope it down.
3. How? – Now I understand the problem, How am I going to solve this?
![](https://www.theschool.ai/wp-content/uploads/2019/02/788fc5e687200665073b7ffd4b47e69bc633b803ede3d853f004b1888bfa7b53.jpg)

### What is dimensionality reduction?
In statistics, machine learning, and information theory, **dimensionality reduction** or **dimension reduction** is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. It can be divided into feature selection and feature extraction.
![](https://www.theschool.ai/wp-content/uploads/2019/02/f773794f52f02db01780418c06625cae97b798053a06eec59e3bbae6a3ac26fc-600x541.jpg)

### Why do we need dimensionality reduction?
Ok, why cannot we use what ever data we get?  That’s what we humans do right? We observe the circumstances, collect data and use our own perception to take decisions. But we easily make mistakes too, if we analyze why most of the times the answer will be simply not enough data. Ok, what if we have too much data? It will confuse us, and we will be forced to take decisions in doubt which will be a disaster.

So even in real life enough data with importance is must to solve a problem. For example what if I ask you to solve the equation x + y = 10 and give values of X,Y,Z?
![](https://www.theschool.ai/wp-content/uploads/2019/02/sry-wrong-dimension.jpg)

The problem of unwanted increase in dimension is closely related to fixation of measuring / recording data at a far granular level then it was done in past. It has started gaining more importance lately due to the surge in data. Particularly ever since the internet of things (IOT) started to get momentum and sensors sends millions of data points using MQTT protocol. We would have a lot of variables/dimensions which are similar and of low (or no) incremental value. This is the problem of high unwanted dimensions and needs a treatment of dimension reduction. With more variables, comes more trouble! And to avoid this trouble, dimension reduction techniques come to the rescue.
![](https://www.theschool.ai/wp-content/uploads/2019/02/monsitj-bigstock-Business-Man-Collect-Data-Into-147442241-600x461.jpg)

### Now the ‘ultima quaestione’, How?
Let’s look at the image shown below. It shows 2 dimensions x1 and x2, which are let us say measurements of several object in cm (x1) and inches (x2). Now, if you were to use both these dimensions in machine learning, they will convey similar information and introduce a lot of noise in system, so you are better of just using one dimension. Here we have converted the dimension of data from 2D (from x1 and x2) to 1D (z1), which has made the data relatively easier to explain.

![](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCADfAOIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9G4fEfiXVNY1y20vTdLNnpt4lmkl3eSo8ubeGXdtWFgP9dt/4DV77V41/6B2g/wDgfN/8ZqPwb/yMHjf/ALDCf+kNpXV0Acx9q8a/9A7Qf/A+b/4zR9q8a/8AQO0H/wAD5v8A4zXT1zVj460a+8Yah4WhuvN1ewt0uJouvyt/X7v/AH0tADftXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6eigDmPtXjX/oHaD/4Hzf/ABmj7V41/wCgdoP/AIHzf/Ga6eigDmPtXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6eigDmPtXjX/oHaD/4Hzf/ABmj7V41/wCgdoP/AIHzf/Ga6eigDmPtXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6eigDmPtXjX/oHaD/4Hzf/ABmj7V41/wCgdoP/AIHzf/Ga6eigDmPtXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6eigDmPtXjX/oHaD/4Hzf/ABmj7V41/wCgdoP/AIHzf/Ga6eigDmPtXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6eigDmPtXjX/oHaD/4Hzf/ABmj7V41/wCgdoP/AIHzf/Ga6eigDmPtXjX/AKB2g/8AgfN/8Zo+1eNf+gdoP/gfN/8AGa6euE17x9INS/sXw7af21rHSX5ysNov96V//ZfvVz1q9PDx5qj327t9kurOTE4qlhIqdV76JbtvsktW/Q1PtXjX/oHaD/4Hzf8Axmo9B17WrjxNcaTq9hYw+XZpcxzWdy8u/c7KysrRrj7tSeFdI1TT4JZNY1T+0r6b5j5caxwxcfdQdcf71EP/ACUif/sDxf8Ao6SrpzdSClKPK+z3/A0o1JVYKUouL7O1/wADp6KKK1NzlPBv/IweN/8AsMJ/6Q2ldXXL+Df+Rg8b/wDYYT/0htK6pulADa8g0P4d6VY/F+9mh8/7baWlvqK3Rb988ks115m9v4lZfl2/7K/3a9fri7D/AJK/rn/YHsv/AEdc0AdpRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHOyeNdFt/FEPhyTUI4tbmg+0Q2sh2u8e7GV9eRVbUviJ4c0jWJtLutTtYb6G3+2SQu/8Aq4t23cfSvL/j8tr4vu9M0DQ7SS/8aQy/abK+s38t9Mx96RpB03bcba5r4G+DdEg8eeIrDxda3Fz47hf7U/8AaD+YlxDuXbcRt91vm/75rxqmPnOq6GFipPq/sryb7+S/A+IxGdYr699RwsYy963O7qK0+BvrPyXzsepx3Gt/E9CLWafQfDH/AD0/1d3eL/s/880/2vvV2vh/w7YeGdP+w6Xax2kA/wCef/oR9TWxRXTh8HGlL2k3zTfV/kl0Xkvnd6n0mGwMaMva1Jc9R/af5JfZXkvnd6hXMQ/8lIn/AOwPF/6Okrp65eP/AJKfP/2CIv8A0dJXoHpHUUUUUDOX8G/8jB43/wCwwn/pDaV1TdK5Xwb/AMjB43/7DCf+kNpXVN0oAbXE2P8AyWDXP+wPZf8Ao64rtq4jTv8Aks2u/wDYDsv/AEdcUAdvRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRUUjiJfMPI/2PmoAloryP4d/tNfD/4k+Cdd8TW2qnRbLw/cTW+s22tj7Jc6c0TNu+0Rt80fTdzVr4f/AB68NfEL4X2fj7Txf2mhXgf7GdQtHgmuVVmVWjjb5mVtu5f72azqVI0ouc3ZIzqVIUYOpUdkt2z0LUtStNJtZLu7uY7aGIfNJK21R9a89uNc1rx15r6Zcnw34YA+fWJiFuLj/rirfKq/7TVzvjzwra/EDwRrep/EPUpfDXhwWcjpFHc/Z/sEWD/pEko/5afd4/hr44+FHjfX/wBoTxj4J8A/F7xHfxfDT97ceHLq4sHsP+E1aKbbb+e+75fl+by/4q7csyavn1CpipT9nQp7qz55K19Nu3waStq7Lmt89KpWx8lz3hSey2lL1f2Yvol7z6tao/Rnw34U0zwlp5h0+1SEkZeXgyynH3mbHzGtn7HD9o87yk87Z5fmbPm2+m6pYYhBCI4+1SVxU6cKUVCmrJdj36VGnRgqdOKUVsuwUUUVobBXLx/8lPn/AOwRF/6OkrqK5eP/AJKfP/2CIv8A0dJQB1FFFFAHKeDf+Rg8b/8AYYT/ANIbSsHxT8SfEtnr82l+G/hxq3iAQ8SapcXNvYWmeu2NpG8yT/eVNv8AtVveDf8AkYPG/wD2GE/9IbSvNf2qNXGg+F9F1aHxXeeG761ubgwfYNKl1GaZWtZlmZYEZfmjjaSTc3yrs+agDtfBPxD8QeINWn0vxD4A1rwrcCPzI7qWW3u7ObttWWKRtrf7MiJVrT/+Swa5/wBgSy/9HXFc18AfCOi+A7XxVonh2/kn0u01RNlhJ5rfYt1pbtt3SMzM0m7zmb+9M1dRZ/8AJYNV/wCwHa/+jrigDs6KKKACiiigAooooAKKKKACiiigAooooAKimn8m3Mh/8c+apa47xh4u/sJYLOyi/tHWrri2sUPXqNzf3UX+9WFatDDwdSo7Jf1Zd2+iOfEYinhabq1XZL+rLu30R+cvxe+E/iP9p/4kXnxX0zwzp/hDRDe2tnYeFNfhnhuPGywThnmvYEb5Y1C/LuXnb83SvuD4I31r8VNB07xzc2FxYDMltYaPc2j2qaasbtGcROqnd8v3v7v3a7bwj4L/ALJuptY1S5/tPxBdIBNcvwkSjny41/hXmu1jrz6dGpiZqriNEto/rLu+qWy83qeXToVcZONfFqyWsYdvOXeXltHzepx/xG+GPhz4seFpvDfirTE1XRJZ4riS1kdgkjROsi7sH7u5fu1X+IPwf8I/E7wzZaB4h0WC80uzniubWFB5f2aSL7jRlfuY6fL613dFe7CvVp8vJNrld1rs3u15s9tpPUhSPyafRRWIwooooAK5eP8A5KfP/wBgiL/0dJXUVy8f/JT5/wDsERf+jpKAOoooooA5Twb/AMjB43/7DCf+kNpXm37UWm2GvaD4c8Py6XrWoa5rV/LY6VJoV9FZSxO1rM0zNLL+72NCsysrI+7P3a9J8G/8jB43/wCwwn/pDaV4x+0R8OfANr4m8O+JfFHhU6jpeoan5euaxI99KthGtvJ5M3lxSbY/3iwx+Zt2rv8A9qgD0L4F+EdQ8F+F72HU9P1C11a7vXubq51TVI7+5vG8uNFlkkjVVX5VVdqr8qxrW3Z/8lh1T/sCWv8A6UTVyH7Ns2iT6B4i/wCEY0p9P8LRaw8emXR+1Yvo/Jh3Tr57M3+s8yPcvyt5ddbb/wDJYNU/7Adp/wClE1AHb0UUUAFFFFABRRRQAUUUUAFFFUbrUrPT7i3gurqGGa6fy4Y5HVWlbrtUd6AL1FUbjUrSxaGO5uYYjNJ5cQkcKXbGdq56muGv9e1Dx1dzaf4dnNrpsXyXWtKuR15SDszcfe/hrlxGIjh0rq8nslu3/W7ei6nHisVHDJJpuT2S3f8AXVvRdWcb8TPizrXgvxNey2F1b3+hxW8Ud1+4Lf2bMzbQzMv39392r/wRsda1Oa+8VakR9i1SNfs32nD3My5/1jN/yzX+7Gtai/BGwm8Q2U17cfavD1hHmz0jZ8n2hv8AWTSt/wAtGb39a2PAXw7k+H2o6rFYam8nh66/eW2mSLu+ySH721v7v+zXytDB46WOWIxV3BN2V/hdt/OP49bJaL0qOHw0sMq2KalibLp7kF/LHvPpKo1rqo2699RRRX2djEKKKKYBRRRQAUUUUAFcvH/yU+f/ALBEX/o6Suorl4/+Snz/APYIi/8AR0lAHUUUUUAcp4N/5GDxv/2GE/8ASG0rgv2jI/Es2n+Fo9Di8S3Wl/2r/wATy18KTJDfva/Z5tu2RmXaqzGFm2/My13vg3/kYPG//YYT/wBIbSvKPi78I7rxn4g1CTVvhX4H+Iuh3UsVxH/aNy9reJIkbRpvV45I5NqtINyunyt92gDv/g3ai38LTxx2vi20/wBIb9140vGuLz7q/dZpJPk/+yrQtf8Akr99/wBgO3/9KJq4v4EeAda+Hcdzp1t4F8H/AA78K+Y9z/ZegXElzNNOdo8xj5cUafKv+3/D92uyt/8Aks1//wBgO3/9KJqAO1ooooAKKKKACiiigAorz1Pirp0XxGu/COoQzaXdiJbizubvCw3y4+byz/s+9WPAvxKsPH99qq6Ta3Eul6fIsSapgfZrtv4vKb+ILjrXHHGUJS5FJc12rdbrf7jz4Y/DVJqnGacm2rdbrf0t5/qjuq8T/aks/CP/AArmfUfE+oPpV7YSebo9/Z5+2Q3vPliEL8zMzY+XvXrGsazY6FYy3eoXUdtbxfflkbbivMtY8Gw/GrUNJvtX0oWui6dI09mJ1/0uVj8v/bNMf8C/3aVbFqjLkhHmm9kvzfZeb+V3obSzCODrxjBOVTdRW/q39mPdv0Sb0PnX4Tpqvxx+J2lf8LT1hI9UtbRbnTdGyY0liHytKgB2tKzfer7d0/T4NNtYba2iSKGNNiRR8Kq1i3ngLw9qFxolzc6RazXGhv5mmybebZtu3KfhXUU6OH9m3VqO83u/0XZL773b3NVHnn9YqO9SS1tol2jFdIrz1b1bbYUUUV2GgUUUUDCiiigAooooAKKKKACuXj/5KfP/ANgiL/0dJXUVy8f/ACU+f/sERf8Ao6SgDqKKKKAOU8G/8jB43/7DCf8ApDaVU+Jl5N/YB0zTNQltNdvD/otraXMMF5dxxsrzJAZfl3eXu+b+Hd/DVvwb/wAjB43/AOwwn/pDaVzfxM+HereL/EnhzUNM1q+0CfTxPGNQ0uG0kmVpNvyyfaI2zF8rf6va27b/AA7qALfgzR/Emka9AbrWZNZ8O/YGEF1cSo0zyNIpVZNq7WKru/eL97+L7u5r8P8AyWa//wCwHb/+lElZ3wbstbsfD99HrsU6Xo1O4/eT2cVp5yqxVZEijZlRW27vvfNu3fLu21pw/wDJYL3/ALAcX/pRJQB2dFFFABRRRQAUUUUAeDfFTwLrXxr8TDw29n/YPh3ST5r6xIitNcyMvyrD/dX5vmar3gPxVf8Aw/8ADMvhzxFoghv9JkSz086ZGFi1Xd91oV/hb+8P4a7rxL44t9EuP7PtYf7S1uUfubCBvn6fef8AuL/tVF4T8JSWN1Jreuyx3/iCZcvKPuW64/1cX91f/Qq+XeH/ANqlPDSvUekpaWS6K3ddFv8AzaHxv1X/AG2VXBTbqu6nJ6xUekbbXX2UtesnbeHSfCd3qmrDWPEpjubiLm1sE+aG0/8Ai3/2q7yiOiveoYeGHi1Hd7t7t93/AF5I+lw+Fp4WLjDVvVt7t92/6tsrIKKKK6jsCiiigYUUUUAFFFFABRRRQAR0+mR0+gArlo/+SnXH/YIi/wDR0ldTXLR/8lOuP+wRF/6OkoA6miiigDkfBv8AyMHjf/sMJ/6Q2ldXXKeDf+Rg8b/9hhP/AEhtK6ugArjI/wDksE//AGA4v/Shq7OuMT/ksE//AGA1/wDShqAOzooooAKKKZQA+vPta8UX2tahcaB4ZWP7RFtF5qgYNFZ5/hUfxvt/h+lQ6hrlz461CfSNBn8nSomaPUNYjP8A31DCf7395v4a7HRdBsdAsIrOxtUtrePny0/mfU15U5zxkvZ0naC3l1flHy7y+S11XhyqzzFuFCXLTWjkt5eUfLvL5LXVUfDPhHT/AAvDKLZGlnnO+e5uHMkszerM1dJT6K9CnThRioU1ZI9ajRp4eCp0lZIZRT6K1NbDKKfRQAyin0UDGUU+igBlFPooAZRT6KAGR0+iigArlo/+SnXH/YIi/wDR0ldTXLR/8lOuP+wRF/6OkoA6miiigDkfBv8AyMHjf/sMJ/6Q2ldXXKeDf+Rg8b/9hhP/AEhtK6ugArjF/wCSwf8AcDX/ANKGrs64z/msH/cD/wDbigDs6K4b4teB7b4geAdW0a51a+0LzI/MXUtPuDDJbunzJIGB6KwDYr4w8M/G7xf8dJ7LwFrXiT+zdE0l3+267oPnRXPijyptkMdpJt+Tcy5dlrixGLhhVepsezhcvjiMLUxk6qhTpfG3f3VbRr+Zt6cq1vbpqfe2qaxZ6HYSXl/cx2sEQ+aWU7VFfOHx0/aAufBF94Yudc0XWdJ+FGoXf2PUvFEA2G0Y7RG0w/1kcEjHb5n/ANavYtL8K3WuX0Gq+JTkw82el798Vv6M5/jk/wBqvO/2r9I+IfxC0rSvhr4K0eCHSvFccttrniy/8ua20i1G3cogb/WSyBtq/T8uf2dTGa1fdh22b9ey8vv6o+M9nVzB3q3jS/l2cv8AF2X91a9+qJ/Dv7RXhu++MmlfCvwDpEniu0tLNrjWda0qZDYaKm3MKvJna7uf4FO75t1e+V8ffs2/Anxf+x/8RT4A0DSv+Eq+EHiAvqMevEQx3+kXqxqrrd/d86Ntq7GX5l+7X2DXpKKilGK0PYjFQSjFWSH0UUVZQUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFctH/yU64/7BEX/o6Suprlo/8Akp1x/wBgiL/0dJQB1NFFFAHI+Df+Rg8b/wDYYT/0htK6uuU8G/8AIweN/wDsMJ/6Q2ldXQAVxn/NYP8AuB/+3FdnXGf81g/7gf8A7cUAVPjD8P7X4ofD/U/Dl2119nugpaG2umt/P2tu8t3XnY33W+tcdcfs26D4m8KWNhrO+C7s3gkspdIc2w07ym3KltgfKBjr/Fn6V7fRXJUwtKtUVSortfd627roHNU56c+d/u3zRV9FL+a3WVtLu+m24xI6fRRXWAUUUUAPoplFAD6KZRQA+imUUAPoplFAD6KZRQA+imUUAPoplFAD6KZWdqmu6fofk/b7uC0N1J5UP2h9u9v7tAGpXLR/8lOuP+wRF/6Okrpq5eP/AJKfP/2CIv8A0dJQB1dFMooA5Twb/wAjB43/AOwwn/pDaV1dcp4N/wCRg8b/APYYT/0htK6ugDip/hbYXFxNMdV15fOk8wrHrFwqj/dXdXHv8M7D/hZ0Fr/aGveT/Y7y+Z/bFxv/ANcvG7du2/7Ney1zh8Oynx8Nb86MQf2e1n5X8e7zFbdQBmf8Kl0n/oIeIf8Awd3X/wAco/4VLpP/AEEPEP8A4O7r/wCOV21FAHE/8Kl0n/oIeIf/AAd3X/xyj/hUuk/9BDxD/wCDu6/+OV21FAHE/wDCpdJ/6CHiH/wd3X/xyj/hUuk/9BDxD/4O7r/45XbUUAcT/wAKl0n/AKCHiH/wd3X/AMco/wCFS6T/ANBDxD/4O7r/AOOV21FAHm954J8LWNxLDc+JdWtZ4o/NeO48R3Csqfd3Nul+7Wivwn0v/oK+If8AweXX/wAcrzbxV4H1Xxp4+sZ9Q8AeZodp/a8ckVxeQSJfTzrHBDNJ825YWt/O+X7y7lXbXc298LD43Q6LFqG2x/4R/wA1dM3jYjLMqrtT+H5aAK3iDwv4W8M3Gh2l/rfiGKfVrz+zrOP+27pmmm8uSTb/AKz+7HI3/Aa2v+FS6T/0EPEP/g7uv/jlcl8VI5Yfjt8F7q5P/Er+16rbf7P2uSxZoT/37juF/wCBV5Domh+Ib/4Ma5pV1p/ji1+Kc0dvFrt/517HDNI2oR+Y1tLu8tl8vcytB92P5fl+7QB9F/8ACpdJ/wCgh4h/8Hd1/wDHKw/C/hXwv4uhvrnS9a8Qyi0vLjTpo5NYu1aKaKTZIpVn/wAqVryK8+Fmt+GvFGt3+j/8JT/xL/G+kf2T5mq3c0Kae62v2zbG0jLJGzSXG7d/7LXpPwJ8y48f/G7UIv3mlXXjBI7WT+Fmi0uwhuNv/baORf8AeVqAOw/4VLpP/QQ8Q/8Ag7uv/jlH/CpdJ/6CHiH/AMHd1/8AHK7aigDif+FS6T/0EPEP/g7uv/jlH/CpdJ/6CHiH/wAHd1/8crtqKAOJ/wCFS6T/ANBDxD/4O7r/AOOUf8Kl0n/oIeIf/B3df/HK7aigDif+FS6T/wBBDxD/AODu6/8Ajlc541+Bseu6R/Z9jq2pRQTSf6TLf6pdTbI/+mce7azf71es0+gDn/B/hWDwXoEOlWt1d3cMI+SS8mMr/wDfTVUj/wCSnz/9giL/ANHSV1dctH/yU64/7BEX/o6SgDpqKfRQB5fo/jzT9B8UeN7W6tdalm/thObPRL26T/jxtP44oWWt3/hamif8+niX/wAJjU//AJHqXwb/AMjB43/7DCf+kNpXV0Acf/wtTRP+fTxL/wCExqf/AMj0f8LU0T/n08S/+Exqf/yPXYUUAcf/AMLU0T/n08S/+Exqf/yPR/wtTRP+fTxL/wCExqf/AMj1e8ceMLTwL4R1TxBdWt3dwafbtcPDYQmaZ1Ufwqv3q5HW/wBofwLoXwxt/iBLrMc3h26RPspgG6a5Zvuwxx/eaTP8NZyqRjpJ2Omlha9ZJ0oN3fLour2R0H/C1NE/59PEv/hMan/8j0f8LU0T/n08S/8AhMan/wDI9b+kapFq2nWV/HFNDHdxrIkVwhR1yu7DKfutXn/xu+Pnh/4A2/hy/wDFlvfRaHq2oLp0+s28O+001mHyyXLf8s42bC7v9qtDnejszof+FqaJ/wA+niX/AMJjU/8A5Ho/4Wpon/Pp4l/8JjU//keua8YftF+EfCvxI8IeBY5bjXvFHiQ+ZBY6Qn2h7e22t/pU23/Vxcfer1WgRx//AAtTRP8An08S/wDhMan/API9H/C1NE/59PEv/hMan/8AI9UNS+KVjofjuDwxqdpcWP2u382y1Gbb9muJOd0QbPDY/vVL4X+J9h4y8R6rpmkWk11Z6aFEuqRj/Rnk/ijVu7LXGsXQc/Z865r2t1va/wCXy8zhjjsNKfs1P3r8tut99vTW+22pa/4Wpon/AD6eJf8AwmNT/wDkeoP+Fi+H/tHm/wBla95//PT/AIRXUd//AH19nrt64j4p/EqL4XeHodavdLvtQ07z1ju5bNd/2SI/emcf3VrtSuejCEqklGKu2RX3j7w1f/Z/tWk69dG1kW4h8zwtqLbJF+6y/wCj/K1W/wDhaOi/8+fiL/wmNT/+R6wNe+PPhuw1Lwtp2lmTxJqfiHbJZ2ul4kf7M33rh/7sa16hRawTpzp251a5x3/C0dF/58/EX/hMan/8j1T0v4geG9JtzDZaXr1rb+Y8vlweFtRVNztuZv8Aj36szM3/AAKp9d+JVh4X8WaXo2pW89rDqEf7jUnAFv5n/PNm/hal0f4jWOv+NL3QNOgnuvskW+5vowPs8Un/ADz3d2rh+uUOf2fMua9rdb2v+Xy8zb6rX5OfkdrXv0te2/r8/Il/4Wpon/Pp4l/8JjU//kej/hamif8APp4l/wDCY1P/AOR68+8F/tX+ENc1jxvpHiMSeAdb8HySyX9j4kdIX+xr929jbdtaFl7rXSfAb412nx+8FnxXpmi6to2hzXkkWny6pD5b30Cn5blF+8Eb/artOY3f+FqaJ/z6eJf/AAmNT/8Akej/AIWpon/Pp4l/8JjU/wD5Hqf4jeLbnwH4H1nXrTRb3xHPp9u1wml6fj7Rcbf4VzXleuftkfD3T/hLpnjqwu5PEH9rP9m0zQ7BN9/d3n/PqIuqyK33t33a7KGDxOKSlQpuSb5dO+9n201u9N9dBOSjuz03/hamif8APp4l/wDCY1P/AOR6P+FqaJ/z6eJf/CY1P/5HrL1z4s2XhS58ORa7YXmk2msxj/S5tvk2k3y4hlI+6x3dfu/Kani+KWlXvxB/4RDTorjU76G3ae8uLdd0FoB91ZH/ALzf3a4rq7V9UcUsbh4zdNzXMmlbrd7et/Ivf8LT0T/nz8S/+Exqf/yPR/wtPRP+fPxL/wCExqf/AMj119RTXEUAzJJHGP8AbO35qZ3HK/8AC09E/wCfPxL/AOExqf8A8j1neHfE1p4k+I17Jaw6hF5WkQhvt+m3Vn1mk+750a7un8Neh1y0f/JTrj/sERf+jpKAOpooooA5Hwb/AMjB43/7DCf+kNpXV1yng3/kYPG//YYT/wBIbSutXpQBx9/4T1W8v55YvF+q2kEr5W3SG2KJ/sruj3VF/wAIRrf/AEPWtf8AgNaf/Ga7OigDyfx5a6r4C8I6pr9z4z8SahBYQPLJbWdjazTP/sqqw18ZWX7P/j/4e3GmfF7U9Okl0n+0JdRuvCdnbpNc6XDLt23EcLL5LTfxMqp8v/oP6S0VyVsOqzTb229e57+WZvUyuM404J8+kr31ja1l29Vr+N/P9L8L61qGnwXcfjnXBFNGsieZaWqvtK5+ZWh+Vq8Z/aytfGun/DmDwr4bh1r4ha54wlbSYbG6sLRtLhUruknvX8n93Eqbv95q+pqK6zwT86f2f/2bvHP7GXxf0rQLqW+8SeHPGNvb2SeMNCsEkuNLu44/+PWfzVdo7T5W2sv+zur7g/4QfW/+h71r/vzaf/Ga7aigR87/ABB+G/in4heJrfwtJqupS+HIUW5vNY1C2tT8/wDCtttjVvM/2qk+FfgHxb4Tm1DwldaxqWlaVp+2TT7/AEuztUtrmI9m3xs3m/3q+hKK8mOXQjifrfM+d77bdvl33PFjlcI4v68pP2j0e3w/y27Lfvfr0OJ/4QTW/wDof9a/782n/wAZrg/i5o/jrQfC4Ph3WNZ8VatfyLZwWsltafZo9/8Ay1n/AHf+rWvcqK9dOx7sJcklJq9j428LfAHxf8AfGGiXWi3c2r6Xrmy21m60yzh+0WM7fxRiRf8Aj23H7tfSv/CB6/8A9FA1r/wGtP8A4zXY0+hu+rN8RiJ4mSnU379zwT4ieBPF3i3VLHwvDrepahpF2PNv9Qv7O18mJFb7se2NW8yo/h38P/FPg7Xr3wq2s6la6IEa50/UrC1tdko3fMszNGzeZXv9FeN/ZkPrSxnM+fbp8P8AL6db731udf8AaE/qv1PlXJ8/iv8AF69LbW0sfnP8UP2SPiL+2d438Ra/4klPhDSvCvm6b4W/tzTrdrvWJY5FZpbxY12/ZGaP5V/2t3+99Kfs/L8QPHPw5gm8WXPiHwN4i0+RtOvNM+x2a22+L5fMtv3XMLfw19CUV7B5R5F8R7LxT4J8E63rWn6/4n8S31rbmSHS9PtLJprluiqv7mvjy1/ZV+Kfwn1Cy+PEMVprfj6W4e51/wAJ6Pp8INvbTbfMaw+Xb9rVUG75Pm3P/wAC/SDy6PLr28uzarlsJ06UU1PSV/tR/l8l1utdtTKVNTab6Hzv8SPCfjPxZpeleHNOv9V1C21uLzL6XVrOz8ixh+XKuFj3eb838P8Adql8PPhf4p+F/jC48KW+qalH4d1CN7221jT7K3L+YNu5blpI2+b+7X0r5dHl18/7Nc/OeVPK4TxaxvO/aLbbSPWPz6t3f5HE/wDCB+IP+iga1/4DWn/xmub8dfDXxBrHh+e1i8R6hr000i7LW9S0ihT/AKaM3k7l2/7PzV61RWp7Rynw78M6t4U8LwafquvXHiC8iH/H1cLyP9n+8f8AgVSxf8lOuP8AsDxf+jpK6auZi/5Kdcf9geL/ANHSUAdTRRRQByPg3/kYPG//AGGE/wDSG0rrV6Vwp8L+K7HxBrV3pGv6Ra2WoXS3Rt7zR5bh42FvDDjet1Hn/U7vu/xVa/sv4gf9DL4a/wDCeuP/AJOoA66iuR/sv4gf9DL4a/8ACeuP/k6j+y/iB/0Mvhr/AMJ64/8Ak6gDrqK5H+y/iB/0Mvhr/wAJ64/+TqP7L+IH/Qy+Gv8Awnrj/wCTqAOuorkf7L+IH/Qy+Gv/AAnrj/5Oo/sv4gf9DL4a/wDCeuP/AJOoA66iuR/sv4gf9DL4a/8ACeuP/k6j+y/iB/0Mvhr/AMJ64/8Ak6gDrqK5H+y/iB/0Mvhr/wAJ64/+TqP7L+IH/Qy+Gv8Awnrj/wCTqAOuqKWXyYDJ/wCgfNXLf2X8QP8AoZfDX/hPXH/ydR/ZfxA/6GXw1/4T1x/8nUAcF4A0eK48da5dReH9d8P3kwluXlvPN+zI7/dmZmkaOWdlb+H/AFartrtfg34gv/Fnww8Oarqkv2q+u7NJJptipvb/AHV6U+bRPHVxGYpfEPhqSKThx/wj1xz/AOT1VNF8HeK/Demw2Gmat4W06wh+5b23huaNE+gW+oA76iuR/sv4gf8AQy+Gv/CeuP8A5Oo/sv4gf9DL4a/8J64/+TqAOuorkf7L+IH/AEMvhr/wnrj/AOTqP7L+IH/Qy+Gv/CeuP/k6gDsaK47+y/iB/wBDL4a/8J64/wDk6j+y/iB/0Mvhr/wnrj/5OoA7GmSVyP8AZfxA/wChl8Nf+E9cf/J1H9l/ED/oZfDX/hPXH/ydQB11Fcj/AGX8QP8AoZfDX/hPXH/ydR/ZfxA/6GXw1/4T1x/8nUAddXMxf8lOuP8AsDxf+jpKr/2X8QP+hl8Nf+E9cf8AydT/AA/4c1y11+fVta1Wwv5ZLVLaKOw057XywGZiW3Ty7vvUAddRRRQB/9k=)

In similar ways, we can reduce n dimensions of data set to k dimensions (k < n) . These k dimensions can be directly identified (filtered) or can be a combination of dimensions (weighted averages of dimensions) or new dimension(s) that represent existing multiple dimensions well.

Simple but very clear explanation, right? Let us thank [Sunil Ray](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/).

Reducing the dimensions of data to 2D or 3D may allow us to plot and visualize it precisely. You can then observe patterns more clearly. Below you can see that, how a 3D data is converted into 2D. First, it has identified the 2D plane then represented the points on these two new axes z1 and z2.

![](https://www.theschool.ai/wp-content/uploads/2019/02/ezgif-2-50606afcba57.jpg)

### Few common methods to perform Dimension Reduction:
* Principal component analysis (PCA) – “simple” linear method that tries to preserve global distances
* Multidimensional scaling (MDS) – tries to preserve global distances
* Sammon’s projection – a variation of the MDS, pays more attention to short distances
* Isometric mapping of data manifolds (ISOMAP) – a graph-based method (of the MDS spirit)
* Curvilinear component analysis (CCA) – MDS-like method that tries to preserve distances in small neighborhoods
* Maximum variance unfolding – maximizes variance with the constraint that the short distances are preserved (an exercise in semidefinite programming)
* Self-organizing map (SOM) – a flexible and scalable method that tries a surface that passes through all data points (originally developed at HUT)
![](https://www.theschool.ai/wp-content/uploads/2019/02/0_3e-wlJFTAbL3YRYV.jpg)

**From this list let us discuss first and last methods now.**

**PCA**

It is one of the most common methods that prevail in the machine learning field for dimensionality reduction. Let us watch [a video by our own Siraj Raval](https://youtu.be/jPmV3j1dAv4) to understand this much better.

PCA is very important in various problems so make sure you understand this well before moving on.

![](https://www.theschool.ai/wp-content/uploads/2019/02/1_cwR_ezx0jliDvVUV6yno5g.jpeg)

**Self-organizing map**

Self Organizing Map(SOM) or Kohonen network by Teuvo Kohonen provides a data visualization technique which helps to understand high dimensional data by reducing the dimensions of data to a map. SOM reduces data dimensions and displays similarities among data.

With SOM, clustering is performed by having several units compete for the current object. Once the data have been entered into the system, the network of artificial neurons is trained by providing information about inputs. The weight vector of the unit is closest to the current object becomes the winning or active unit. During the training stage, the values for the input variables are gradually adjusted in an attempt to preserve neighborhood relationships that exist within the input data set. As it gets closer to the input object, the weights of the winning unit are adjusted as well as its neighbors.
```
Teuvo Kohonen writes “The SOM is a new, effective software tool for the visualization of high-dimensional data. It converts complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display. As it thereby compresses information while preserving the most important topological and metric relationships of the primary data items on the display, it may also be thought to produce some kind of abstractions.”
```
**The Algorithm:**
1. Each node’s weights are initialized.
2. A vector is chosen at random from the set of training data.
3. Every node is examined to calculate which one’s weights are most like the input vector. The winning node is commonly known as the Best Matching Unit (BMU).
4. Then the neighborhood of the BMU is calculated. The amount of neighbors decreases over time.
5. The winning weight is rewarded with becoming more like the sample vector. The neighbors also become more like the sample vector. The closer a node is to the BMU, the more its weights get altered and the farther away the neighbor is from the BMU, the less it learns.
6. Repeat from step 2 for N iterations.

**Best Matching Unit ( BMU)** is a technique which calculates the distance from each weight to the sample vector, by running through all weight vectors. The weight with the shortest distance is the winner.

![](https://www.theschool.ai/wp-content/uploads/2019/02/brace-yourself-implementation-is-coming.jpg)

In general, SOMs might be useful for visualizing high-dimensional data in terms of its similarity structure. Especially large SOMs (i.e. with large number of Kohonen units) are known to perform mappings that preserve the topology of the original data, i.e. neighboring data points in input space will also be represented in adjacent locations on the SOM.

The following code shows the ‘classic’ color mapping example, i.e. the SOM will map a number of colors into a rectangular area.

Install the following to get started ( for linux machines)
```
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
pip install pymvpa2
```
Now you should be able to import this module in a jupyter notebook.
```
from mvpa2.suite import *
```
First, we define some colors as RGB values from the interval (0,1), i.e. with white being (1, 1, 1) and black being (0, 0, 0). Please note, that a substantial proportion of the defined colors represent variations of ‘blue’, which are supposed to be represented in more detail in the SOM.
```
colors = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

# store the names of the colors for visualization later on
color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
         'greyblue', 'lilac', 'green', 'red',
         'cyan', 'violet', 'yellow', 'white',
         'darkgrey', 'mediumgrey', 'lightgrey']
```
Now we can instantiate the mapper. It will internally use a so-called Kohonen layer to map the data onto. We tell the mapper to use a rectangular layer with 20 x 30 units. This will be the output space of the mapper. Additionally, we tell it to train the network using 400 iterations and to use custom learning rate.
```
som=SimpleSOMMapper((20,30),400,learning_rate=0.05)
```
Finally, we train the mapper with the previously defined ‘color’ dataset.
```
som.train(colors)
```
Each unit in the Kohonen layer can be treated as a pointer into the high-dimensional input space, that can be queried to inspect which input subspaces the SOM maps onto certain sections of its 2D output space. The color-mapping generated by this example’s SOM can be shown with a single matplotlib call:
```
pl.imshow(som.K, origin='lower')
```
And now, let’s take a look onto which coordinates the initial training prototypes were mapped to. The get those coordinates we can simply feed the training data to the mapper and plot the output.
```
mapped = som(colors)

pl.title('Color SOM')
# SOM's kshape is (rows x columns), while matplotlib wants (X x Y)
for i, m in enumerate(mapped):
    pl.text(m[1], m[0], color_names[i], ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.5, lw=0))
```
The text labels of the original training colors will appear at the ‘mapped’ locations in the SOM – and should match with the underlying color.

The following figure shows an exemplary solution of the SOM mapping of the 3D color-space onto the 2D SOM node layer:

![](https://www.theschool.ai/wp-content/uploads/2019/02/ex_som.png)

Great Keep learning! All the Best. Meet you guys next week.

Credits:

http://www.pymvpa.org/examples/som.html

https://research.cs.aalto.fi/pml/papers/kaski11spm.pdf

https://github.com/hammadshaikhha/Math-of-Machine-Learning-Course-by-Siraj/blob/master/Self%20Organizing%20Maps%20for%20Data%20Visualization/Self%20Organizing%20Map%20for%20Clustering.ipynb

http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html