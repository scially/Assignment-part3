import sklearn.tree
import streamlit as st
import pandas as pd
import numpy as np 
import sklearn 

@st.cache_data
def get_dji_data()->pd.DataFrame:
    print('-> Get DJI Dataset from dji.pickle')
    dji_data: pd.DataFrame = pd.read_pickle('dji.pickle')
    print(dji_data)
    return dji_data

@st.cache_data
def get_rate_data()->pd.DataFrame:
    print('-> Get Rate Dataset from rate.pickle')
    rate_data: pd.DataFrame = pd.read_pickle('rate.pickle')
    print(rate_data)
    print('--> Delete Missing Value and Format Date')
    rate_data['time'] = pd.to_datetime(rate_data['time'],format='%Y年%m月%d日 ')
    print(rate_data)
    return rate_data

@st.cache_data
def get_unemployment_data()->pd.DataFrame:
    print('-> Get Unemployment Dataset from unemployment.pickle')
    unemployment_data: pd.DataFrame = pd.read_pickle('unemployment.pickle')
    print(unemployment_data)
    return unemployment_data

@st.cache_data
def get_rate_data_by_year()->pd.DataFrame:
    rate_data = get_rate_data()
    print('-> process rate_data group by year and take mean value')
    rate_data['year'] = rate_data['time'].dt.year
    rate_data_year = rate_data.groupby('year').agg('mean')
    rate_data_year = rate_data_year.drop(columns=['time'])
    print(rate_data_year)
    return rate_data_year

@st.cache_data
def get_unemployment_data_by_year()->pd.DataFrame:
    unemployment_data = get_unemployment_data()
    print('-> process unemployment_data group by year and take mean value')
    unemployment_data = unemployment_data.rename(columns={'Unnamed: 0':'time'})
    unemployment_data['time'] = pd.to_datetime(unemployment_data['time'], format='%Y-%m-%d')
    unemployment_data['year'] = unemployment_data['time'].dt.year
    unemployment_data_year = unemployment_data.groupby('year').agg('mean')
    unemployment_data_year = unemployment_data_year.drop(columns=['time'])
    print(unemployment_data_year)
    return unemployment_data_year

@st.cache_data
def get_dji_data_by_year()->pd.DataFrame:
    dji_data = get_dji_data()
    print('-> process dji_data group by year and take mean value')
    dji_data['year'] = dji_data.index.year
    dji_data_year = dji_data.groupby('year').agg('mean')
    print(dji_data_year)
    return dji_data_year

st.set_page_config(
    page_title="LASTNAME_FIRSTNAME_part3",
    page_icon="👋",
)

def intro():
    import streamlit as st

    st.write("# Welcome to LASTNAME_FIRSTNAME_part3! 👋")
    st.markdown('# Name: YingZhou')
    st.markdown('''# 程序介绍
## The data shows
1. Data Source 1(https://www.investing.com/central-banks/fed-rate-monitor) Real-time and historical interest rate changes from the Federal Reserve in the past and future. As an important tool for macroeconomic control, the Federal Reserve's interest rate has an important impact on the global economy. By adjusting these interest rates, the Fed can effectively manage economic growth, control inflation, influence employment levels, and thereby influence global finance.
2. Dataset 2 (Yahoo Finance Public API) provides an analysis of the DJI index on the performance of the U.S. stock market. It can be accessed through Yahoo Finance’s external public API and can be used to investigate the impact of the Federal Reserve’s interest rate adjustments on stock market behavior.
3. Data 3 represents the unemployment rate, which represents the percentage of unemployed people in the labor force. Labor force data are limited to persons 16 years of age and older who currently reside in one of the 50 states or the District of Columbia and who do not reside in institutions (such as prisons and mental hospitals, nursing homes), and persons who are not serving in the armed forces. This ratio is also defined as the U-3 indicator of labor underutilization.
## Program module description
1. About General introduction to the program
2. General introduction to the Raw Data program
3. Explore to view the original data of all data, you can use the slidebar to view the year interval data you want to pay attention to
4. U.S. state unemployment rates perform join correlation analysis on the data, and use pandas to calculate the average unemployment rate of each state, the DJI average index and the Federal Reserve average interest rate. You can interact in this module. First, you can view the data of the year of interest through the slidebar. Also click on the icon to view precise data
5. Regression Analysis uses sklearn for data modeling, using the unemployment rate as the dependent variable, the Federal Reserve interest rate and DJI as independent variables, conducts multiple model regression analyses, and selects the model with the best fitting effect. Through this module we can produce There is a certain correlation between the unemployment rate and the Federal Reserve and DJI.
## Work that can continue to be improved
1. The data can be modeled more accurately, including using neural networks to mine more years of data and accurately analyze the unemployment rate.
''')
    st.sidebar.success("Select a part above.")

def more_intro():
    import streamlit as st
    st.write("# Welcome to LASTNAME_FIRSTNAME_part3! 👋")
    st.markdown('# More detailed instructions')
    st.markdown('''1. The focus of the project is to analyze the relationship between the U.S. unemployment rate and DJI and Federal Reserve interest rates through crawlers and machine learning.
2. The conclusion is that there is a certain relationship between the U.S. unemployment rate and DJI and Federal Reserve interest rates, and the initial hypothesis has been confirmed
3. Encountered difficulties in Python implementation, including how sklearn models and how python crawls to obtain information.
4. Master the basic capabilities of data mining based on Python
5. Learn neural networks for more precise analysis''')

def show_raw_table():
    rate_data = get_rate_data()
    dji_data = get_dji_data()
    unemployment_data = get_unemployment_data()
    
    if st.checkbox('Show raw rate data'):
        st.subheader('rate data')
        st.write(rate_data)

    if st.checkbox('Show raw unemployment data'):
        st.subheader('unemployment data')
        st.write(unemployment_data)

    if st.checkbox('show raw dji data'):
        st.subheader("dji data")
        st.write(dji_data)

def data_explore():
    import altair
    
    unemployment_data_year = get_unemployment_data_by_year()
    rate_data_year = get_rate_data_by_year()
    dji_data_year = get_dji_data_by_year()
    all_data = unemployment_data_year.join(rate_data_year, how='inner')
    all_data = all_data.join(dji_data_year, how='inner')
    
    start_year, end_year = st.select_slider(
    'Select a range of year',
    options=all_data.index,
    value=(all_data.index[0], all_data.index[-1]))
   
    rate_line = (altair
         .Chart(all_data[(all_data.index >= start_year) & (all_data.index <= end_year)].reset_index())
         .mark_line()
         .encode(x='year', y='rate'))
    st.altair_chart(rate_line, use_container_width=True)
    
    dji_open_line = (altair
                .Chart(all_data[(all_data.index >= start_year) & (all_data.index <= end_year)].reset_index())
                .mark_line()
                .encode(x='year', y=altair.Y('Open', title='DJI')))
    
    st.altair_chart(dji_open_line, use_container_width=True)

def data_all_state_unemployment_explore():
    import altair as alt
    
    unemployment_data_year = get_unemployment_data_by_year()
    rate_data_year = get_rate_data_by_year()
    dji_data_year = get_dji_data_by_year()
    all_data = unemployment_data_year.join(rate_data_year, how='inner')
    all_data = all_data.join(dji_data_year, how='inner')
    
    select_year = st.select_slider(
    'Select year of the data',
    options=(all_data.index))
    
    all_data_year = all_data[all_data.index == select_year].reset_index()
    dji_data_year = dji_data_year[dji_data_year.index == select_year].reset_index()
    unemployment_data_year =  unemployment_data_year[unemployment_data_year.index == select_year].reset_index()
    st.write(f"rate: {all_data_year['rate'][0]}%")
    st.write(f"dji: {dji_data_year['Open'][0]}")
    unemployment_data_year_t = unemployment_data_year.transpose().reset_index()
    unemployment_data_year_t = unemployment_data_year_t.rename(columns={
        'index':'state',
        0: 'unemployment'
    })
    unemployment_data_year_t = unemployment_data_year_t.drop([0])
    b = (alt
         .Chart(unemployment_data_year_t)
         .mark_bar()
         .encode(x='state', y='unemployment'))
    
    st.altair_chart(b, use_container_width=True)

def regression_analysis():
    import altair as alt
    from lazypredict.Supervised import LazyRegressor
    
    unemployment_data_year = get_unemployment_data_by_year()
    rate_data_year = get_rate_data_by_year()
    dji_data_year = pd.DataFrame(get_dji_data_by_year()['Open'])
    dji_data_year = dji_data_year.rename(columns={
        'Open': 'dji'
    })
    
    unemployment_data_year = pd.DataFrame(unemployment_data_year.mean(axis=1))
    unemployment_data_year = unemployment_data_year.rename(columns={
        0: 'unemployment'
    })
    
    all_data = unemployment_data_year.join(rate_data_year, how='inner')
    all_data = all_data.join(dji_data_year, how='inner')
    
    x = all_data['unemployment']
    y = all_data[['rate','dji']]
    
    percetange = st.select_slider(
    'Select percentage of the perticate data and train data',
    options=np.linspace(0.7, 1.0, 100))
    
    if st.button('Start Train'):
        data_load_state = st.text('Start train data...')
        
        offset = int(x.shape[0] * percetange)
        X_train, y_train = pd.DataFrame(x[:offset]), y.iloc[:offset]
        X_test, y_test = pd.DataFrame(x[offset:]), y.iloc[offset:]
        reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        
        st.subheader('Tran Predictions')
        st.write(predictions)
        
        data_load_state.text('Train Successfully')

        st.write('Use ExtraTreeRegressor to Predict')
        reg = sklearn.tree.ExtraTreeRegressor()
        reg.fit(pd.DataFrame(x), y)
        y_pre = reg.predict(pd.DataFrame(x))
        
        y_pre = pd.DataFrame(y_pre)[0]
        raw_line = (alt.Chart(all_data.reset_index())
                    .mark_line()
                    .encode(x='year', y='unemployment'))
        
        y_pre = pd.concat((all_data.reset_index()['year'], y_pre), axis=1)
        pre_line = (alt.Chart(y_pre)
                    .mark_line(color='red')
                    .encode(x='year', y=alt.Y('0', title='predict_unemployment')))
        
        st.altair_chart(raw_line + pre_line, use_container_width=True)
        
    
page_names_to_funcs = {
    "About": intro,
    'More detailed instructions': more_intro,
    "Raw Data": show_raw_table,
    "Explore": data_explore,
    "U.S. state unemployment rates": data_all_state_unemployment_explore,
    'Regression Analysis': regression_analysis
}

page_name = st.sidebar.selectbox("Choose a type", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()