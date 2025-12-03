from statistical_tools import StatisticalTools

csv_path = 'C:/Users/xylop/Documents/github_repos/LAP_BLACK/weather_historic/magiko/magiko_total.csv'

cdd = StatisticalTools(csv_path=csv_path, date_col='DATE', date_format='%Y-%m-%d', precip_col='Rain')
df_with_classes, thresholds = cdd.analyze_precip_extremes()
pe_counts = cdd.count_extreme_precipitation_events_by_year(threshold=thresholds['upper'])
cdd_lengths, cdf, inv_cdf = cdd.analyze_cdd_distribution()
spid = cdd.compute_spi()
print()
