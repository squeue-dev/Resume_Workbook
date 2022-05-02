#!/usr/bin/env python
# coding: utf-8

# <h1> NIST's CCCBDB Geometries Scraper </h1>
# 
# Description: Web scraper designed to obtain the corresponding Z-matrix data for every molecule available in NIST's Computational Chemistry Comparison and Benchmark Data Base.
# 
# <ul>
#     <li> Authors: Alfonso Esqueda García </li>
# </ul>

from os import listdir
from os.path import isfile, join

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException

from tqdm import tqdm

import time


# In[4]:


# First of all let's get a pandas' DataFrame that contains the formulas of every molecule in the CCCBDB
#df_list = pd.read_html('https://cccbdb.nist.gov/listallnumx.asp')
#df = df_list[-1]
#df.to_csv('../CSVs/CCCBDB_listallnum.csv', index=False)

df = pd.read_csv('../CSVs/CCCBDB_listallnum.csv')
df.head(10)


# In[5]:


# Drop Rows which separate species by number of atoms

df.drop(df.loc[df['link'] == df['species']].index, inplace=True)
df

options = Options()
options.headless = True

for i in tqdm(range(1550, len(df.index))):
    formula = str(df['species'].values[i])

    donfs = [f for f in listdir('../Geoms/Raw_HTMLs/') if isfile(join('../Geoms/Raw_HTMLs/', f))]
    
    html_file = formula + '.html'
    if html_file in donfs:
        print('Geom file already in folder\n')
        continue

    driver = webdriver.Firefox(options=options)
    driver.get('https://cccbdb.nist.gov/mdlmol1x.asp')
    try:
        assert 'CCCBDB' in driver.title
    except AssertionError:
        print('CCCBDB not in title\n')
        driver.close()
        continue

    text1 = driver.find_element_by_id('text1')
    text1.send_keys(formula)

    submit1 = driver.find_element_by_id('submit1')
    submit1.click()
    
    # In some cases the CCCBDB asks to select one species for the formula queried...
    # In that case the ground state should be selected.
    if 'https://cccbdb.nist.gov/getonex.asp' == driver.current_url:
        try:
            driver.find_element_by_xpath("//input[@value='1']").click()
            driver.find_element_by_xpath("//input[@type='submit']").click()
        except NoSuchElementException:
            print('Not submit\n')
            driver.close()
            continue

    url = 'mdlmol3x.asp?method=11&basis=6'
    try:
        driver.find_element_by_xpath('//a[@href="'+url+'"]').click()
    except NoSuchElementException:
        print('Not basis\n')
        driver.close()
        continue

    filename = '../Geoms/Raw_HTMLs/' + formula + '.html'
    f = open(filename, 'w')
    f.write(driver.page_source)
    f.close()

    driver.close()

    time.sleep(3)
