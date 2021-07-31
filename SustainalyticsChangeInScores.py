

import pandas as pd
import numpy as np

hist_raw_scores_df = pd.read_csv(r'C:\Users\rol\Desktop\w266\FinalProject\HistoricalWeightedScores.csv',
                                 encoding="ISO-8859-1")

#Check which permanent ID to use
UNIQUE_ID_missing = hist_raw_scores_df[hist_raw_scores_df.UNIQUE_ID.isna()]
print("# of missing unique id's:", len(UNIQUE_ID_missing))
CapitalIQ_ID_missing = hist_raw_scores_df[hist_raw_scores_df.CapitalIQ_ID.isna()]
print("# of missing CapitalIQ id's:", len(CapitalIQ_ID_missing))

#Convert this column to datetime
hist_raw_scores_df['Date'] = pd.to_datetime(hist_raw_scores_df['Date'].astype(str), format='%Y%m%d')

#Sort by Id and date and calculate leads and lags
hist_raw_scores_df = hist_raw_scores_df.sort_values(by=['CapitalIQ_ID', 'Date'])
hist_raw_scores_df['Date_lead1'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])['Date'].shift(-1)
hist_raw_scores_df['Date_lead2'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])['Date'].shift(-2)
hist_raw_scores_df['Date_lead3'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])['Date'].shift(-3)
hist_raw_scores_df['Date_diff_lead1'] = (hist_raw_scores_df['Date_lead1'] - hist_raw_scores_df['Date']).dt.days
hist_raw_scores_df['Date_diff_lead2'] = (hist_raw_scores_df['Date_lead2'] - hist_raw_scores_df['Date']).dt.days
hist_raw_scores_df['Date_diff_lead3'] = (hist_raw_scores_df['Date_lead3'] - hist_raw_scores_df['Date']).dt.days


#The intervals aren't regular so look at the distribution for cutoffs; obviously
#want to cutoff as few as possible but still remove bad data
hist_raw_scores_df['Date_diff_lead1'].describe(percentiles=[.05, .10, .25, .50, .75, .90, .95, .99])
hist_raw_scores_df['Date_diff_lead2'].describe(percentiles=[.05, .10, .25, .50, .75, .90, .95, .99])
hist_raw_scores_df['Date_diff_lead3'].describe(percentiles=[.05, .10, .25, .50, .75, .90, .95, .99])

#Before removing bad data we have nan's from the last date. Count them.
print("# of missing lead1 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Date_lead1'].isna()]))
print("# of missing lead2 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Date_lead2'].isna()]))
print("# of missing lead3 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Date_lead3'].isna()]))

#Set the acceptable ranges along with removing nulls
hist_raw_scores_df['Lead1_Usable'] = np.where((hist_raw_scores_df['Date_diff_lead1'] > 58) |
                                              (hist_raw_scores_df['Date'].isna()) |
                                              (hist_raw_scores_df['Date_lead1'].isna()) , False, True)
hist_raw_scores_df['Lead2_Usable'] = np.where((hist_raw_scores_df['Date_diff_lead2'] < 46) |
                                                 (hist_raw_scores_df['Date_diff_lead2'] > 88) |
                                                 (hist_raw_scores_df['Date'].isna()) |
                                                 (hist_raw_scores_df['Date_lead2'].isna()), False, True)
hist_raw_scores_df['Lead3_Usable'] = np.where((hist_raw_scores_df['Date_diff_lead3'] < 72) |
                                                 (hist_raw_scores_df['Date_diff_lead3'] > 118) |
                                                 (hist_raw_scores_df['Date'].isna()) |
                                                 (hist_raw_scores_df['Date_lead3'].isna()), False, True)



#Check after removing potentially bad dates
print("# of missing lead1 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Lead1_Usable']==False]))
print("# of missing lead2 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Lead2_Usable']==False]))
print("# of missing lead3 dates before processing", len(hist_raw_scores_df[hist_raw_scores_df['Lead3_Usable']==False]))


#Not much of an increase in missing observations and the date diffs seem reasonable
#so we will proceed using these

esg_cols_temp = [col for col in hist_raw_scores_df.columns if col.startswith('E_') |
            col.startswith('S_') | col.startswith('G_') ]

esg_cols = ['total_esg_score','governance_score','social_score','environment_score']
esg_cols.extend(esg_cols_temp)


for col in esg_cols:
    hist_raw_scores_df[col+'_lead_1'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])[col].shift(-1)
    hist_raw_scores_df[col+'_lead_2'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])[col].shift(-2)
    hist_raw_scores_df[col+'_lead_3'] = hist_raw_scores_df.groupby(['CapitalIQ_ID'])[col].shift(-3)
    hist_raw_scores_df[col+'_month_change_pure'] = np.where(hist_raw_scores_df['Lead1_Usable'],
                                                           hist_raw_scores_df[col+'_lead_1'] -
                                                           hist_raw_scores_df[col],
                                                           np.nan) 
    hist_raw_scores_df[col+'_month_change_pct'] = np.where(hist_raw_scores_df['Lead1_Usable'],
                                                           hist_raw_scores_df[col+'_lead_1']/
                                                           hist_raw_scores_df[col]-1,
                                                           np.nan) 
    hist_raw_scores_df[col+'_quarter_change_pure'] = np.where(hist_raw_scores_df['Lead3_Usable'],
                                                           hist_raw_scores_df[col+'_lead_3'] -
                                                           hist_raw_scores_df[col],
                                                           np.nan) 
    hist_raw_scores_df[col+'_quarter_change_pct'] = np.where(hist_raw_scores_df['Lead3_Usable'],
                                                           hist_raw_scores_df[col+'_lead_3']/
                                                           hist_raw_scores_df[col]-1,
                                                           np.nan) 

hist_raw_scores_df['all_month_change_pure'] = hist_raw_scores_df[[col for col in hist_raw_scores_df.columns \
                                                                  if col.endswith('month_change_pure')]].sum(axis=1)
hist_raw_scores_df['all_quarter_change_pure'] = hist_raw_scores_df[[col for col in hist_raw_scores_df.columns \
                                                                  if col.endswith('quarter_change_pure')]].sum(axis=1)


#Check monthly distributions
dist_monthly_df = hist_raw_scores_df[[col for col in hist_raw_scores_df.columns if col.endswith('month_change_pure')]]
dist_monthly_df_summary = dist_monthly_df.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
dist_monthly_df_summary = dist_monthly_df_summary[['all_month_change_pure']+
                                                   [col for col in dist_monthly_df_summary.columns if col != 'all_month_change_pure']]
dist_monthly_df_summary.to_csv(r'C:\Users\rol\Desktop\w266\FinalProject\MonthlyDistScoresWeighted.csv')


#Check quarterly distributions
dist_quarterly_df = hist_raw_scores_df[[col for col in hist_raw_scores_df.columns if col.endswith('quarter_change_pure')]]
dist_quarterly_df_summary = dist_quarterly_df.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
dist_quarterly_df_summary = dist_quarterly_df_summary[['all_quarter_change_pure']+
                                                   [col for col in dist_quarterly_df_summary.columns if col != 'all_quarter_change_pure']]
dist_quarterly_df_summary.to_csv(r'C:\Users\rol\Desktop\w266\FinalProject\QuarterlyDistScoresWeighted.csv')


quarterly_change = hist_raw_scores_df[['Date', 'CapitalIQ_ID', 'Ticker', 
                                       'Company', 'total_esg_score', 
                                       'total_esg_score_lead_1', 'total_esg_score_lead_3',
                                       'total_esg_score_month_change_pure',
                                       'total_esg_score_month_change_pct',
                                       'total_esg_score_quarter_change_pure',
                                       'total_esg_score_quarter_change_pct',
                                       'governance_score_lead_1', 'governance_score_lead_3',
                                       'governance_score_month_change_pure',
                                       'governance_score_month_change_pct',
                                       'governance_score_quarter_change_pure',
                                       'governance_score_quarter_change_pct',
                                       'social_score_lead_1', 'social_score_lead_3',
                                       'social_score_month_change_pure',
                                       'social_score_month_change_pct',
                                       'social_score_quarter_change_pure',
                                       'social_score_quarter_change_pct',
                                       'environment_score_lead_1', 'environment_score_lead_3',
                                       'environment_score_month_change_pure',
                                       'environment_score_month_change_pct',
                                       'environment_score_quarter_change_pure',
                                       'environment_score_quarter_change_pct']].copy()

quarterly_change.to_csv(r'C:\Users\rol\Desktop\w266\FinalProject\sustainalytics_scores.csv')

#Manually check a large change or two
large_df = hist_raw_scores_df[hist_raw_scores_df.E_1_1_quarter_change_pure==100]\
    [['Date', 'CapitalIQ_ID', 'Ticker', 'Company', 'E_1_1', 'E_1_1_lead_3', 
     'E_1_1_quarter_change_pure']]

one_large_df = hist_raw_scores_df[hist_raw_scores_df.CapitalIQ_ID=='IQ1038328']\
    [['Date', 'CapitalIQ_ID', 'Ticker', 'Company', 'E_1_1', 'E_1_1_lead_3', 
     'E_1_1_quarter_change_pure']]



map_df = hist_raw_scores_df[['Date', 'CapitalIQ_ID', 'Ticker']]
unique_tickers = pd.DataFrame(map_df.Ticker.unique())
unique_tickers.rename(columns={0:'Ticker'}, inplace=True)
unique_tickers.to_csv(r'C:\Users\rol\Desktop\w266\FinalProject\UniqueTickers.csv',
                      index=False)
