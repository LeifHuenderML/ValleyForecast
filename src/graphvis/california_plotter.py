import plotly.express as px
import pandas as pd
import json
import numpy as np

class California_Plot:
    def __init__(self, df, counties, title, locations='county', color='Cases'):
        self.df = df
        self.counties = counties
        self.title = title
        self.color = color
        self.locations = locations
        self.fig = None

    def plot(self):
        bins = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 
                1000, 2000, 3000, 4000, 5000, 40000, 50000]
        
        labels = ['0/NaN', '1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', 
                  '71-80', '81-90', '91-100', '101-200', '201-300', '301-400', '401-500',
                  '501-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', 
                  '5001-40000', '40001-50000']
        
        self.df['  Cases'] = pd.cut(self.df[self.color], bins=bins, labels=labels)
        
        discrete_colors = ['#d8dee9', '#e50000', '#dc0000', '#d40000', '#cc0000', '#c40000', '#bc0000', '#b40000', '#ac0000', '#a40000', '#9c0000', '#940000', '#8b0000', '#830000', '#7b0000', '#730000', '#6b0000', '#630000', '#5b0000', '#530000', '#4b0000', '#430000', '#3b0000']
        color_scale = dict(zip(labels, discrete_colors))

        self.fig = px.choropleth(
            self.df,
            geojson=self.counties,
            locations=self.locations,
            color='  Cases',
            color_discrete_map=color_scale,
            featureidkey="properties.NAME",
            projection='mercator',
            labels={self.color: 'Cases'}
        )

        self.fig.update_geos(
            fitbounds="locations",
            visible=False,
            # bgcolor='#d8dee9',
            projection_scale=7,
            center={"lat": 37.5, "lon": -119.5},
            lataxis_range=[32.5, 42.0],
            lonaxis_range=[-125, -113.5]
        )

        self.fig.update_layout(
            title_text=self.title,
            # showlegend=False,
            # paper_bgcolor='#d8dee9',
            # plot_bgcolor='#d8dee9',
            title_x=0.5
        )

        self.fig.update_layout(legend=dict(
            y=0.99,
            x=0.70
        ))

        self.fig.update_layout(
            height=800,
            width=800,
        )



        self.fig.show()

    def save(self, file_path):
        if self.fig:
            self.fig.write_html(file_path)

