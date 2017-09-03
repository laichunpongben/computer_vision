import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class LabelHandler(object):
    def __init__(self, path):
        self.path = path
        self.df = self.get_df()

    def get_df(self):
        df = pd.read_csv(self.path)
        df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
        df = df[['Subject', 'Zone', 'Probability']]
        return df

    def get_hit_rate_stats(self):
        df_summary = self.df.groupby('Zone')['Probability'].agg(['sum','count'])
        df_summary['Zone'] = df_summary.index
        df_summary['pct'] = df_summary['sum'] / df_summary['count']
        df_summary.sort_values('pct', axis=0, ascending= False, inplace=True)

        return df_summary

    @staticmethod
    def chart_hit_rate_stats(df_summary):
        fig, ax = plt.subplots(figsize=(15,5))
        sns.barplot(ax=ax, x=df_summary['Zone'], y=df_summary['pct']*100)

    @staticmethod
    def print_hit_rate_stats(df_summary):
        print ('{:6s}   {:>4s}   {:6s}'.format('Zone', 'Hits', 'Pct %'))
        print ('------   ----- ----------')
        for zone in df_summary.iterrows():
            print ('{:6s}   {:>4d}   {:>6.3f}%'.format(zone[0], np.int16(zone[1]['sum']), zone[1]['pct']*100))
        print ('------   ----- ----------')
        print ('{:6s}   {:>4d}   {:6.3f}%'.format('Total ', np.int16(df_summary['sum'].sum(axis=0)),
                                                 ( df_summary['sum'].sum(axis=0) / df_summary['count'].sum(axis=0))*100))

    def get_subject_labels(self, subject_id):
        # Separate the zone and subject id into a df
        threat_list = self.df.loc[self.df['Subject'] == subject_id]

        return threat_list

    def get_zone_labels(self, zone):
        zone_labels = self.df.loc[self.df['Zone'] == 'Zone'+str(zone+1)]
        return np.array(zone_labels['Probability'].tolist())

    def get_subject_ids(self):
        return sorted(list(set(self.df['Subject'].tolist())))

if __name__ == '__main__':
    THREAT_LABELS = '/media/ben/Data/kaggle/passenger_screening_dataset/stage1/stage1_labels.csv'
    label = LabelHandler(THREAT_LABELS)
    df_summary = label.get_hit_rate_stats()
    label.chart_hit_rate_stats(df_summary)
    label.print_hit_rate_stats(df_summary)

    # subject_id = '00360f79fd6e02781457eda48f85da90'
    # print(label.get_subject_labels(subject_id))
    # plt.show()

    subject_ids = label.get_subject_ids()
    # print(subject_ids)
    print(len(subject_ids))

    zone = 0
    zone_labels = label.get_zone_labels(zone)
    print(zone_labels)
    print(zone_labels.shape)
