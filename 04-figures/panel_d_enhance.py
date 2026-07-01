import json, numpy as np, pandas as pd
DATA='/Volumes/PARTA/STRIGA_ANALYSIS/climate_projections_publication/outputs/panel_data_corrected'
pa=json.load(open(f'{DATA}/panel_a_trajectories.json'))
pc=pd.read_csv(f'{DATA}/panel_c_phase_space.csv')
existing=json.load(open(f'{DATA}/panel_d_regional_signatures.json'))
CR={'Burkina Faso':'West Africa','Ghana':'West Africa','Mali':'West Africa','Nigeria':'West Africa',
    'Ethiopia':'East Africa','Kenya':'East Africa','Tanzania':'East Africa','Uganda':'East Africa',
    'Zambia':'Southern Africa'}
metrics=['Production Growth','Loss Rate Change','Technology Adoption','Control Coverage','Climate Impact']
regions=['West Africa','East Africa','Southern Africa']

def pcv(region):
    rc=[c for c,r in CR.items() if r==region]
    cv={m:[] for m in metrics}
    for c in rc:
        cd=pc[pc['country']==c]
        if not cd.empty: cv['Production Growth'].append((c, min(1.0, cd.iloc[0]['production_growth_pct']/100)))
    for c in rc:
        cd=pc[(pc['country']==c)&(pc['scenario']=='Continued Trends')]
        if not cd.empty: cv['Loss Rate Change'].append((c, max(0,min(1,(cd['loss_rate_change_pp'].values[0]+5)/10))))
    for c in rc:
        b=pa['baseline_2020'].get(c,{}).get('mean_millions',100)
        ig=pa['projections_2050'].get(c,{}).get('Integrated Management',{}).get('mean_millions',b)
        if b>0: cv['Technology Adoption'].append((c, max(0,(b-ig)/b)))
    for c in rc:
        b=pa['baseline_2020'].get(c,{}).get('mean_millions',100)
        en=pa['projections_2050'].get(c,{}).get('Enhanced Control',{}).get('mean_millions',b)
        co=pa['projections_2050'].get(c,{}).get('Continued Trends',{}).get('mean_millions',b)
        if (co-b)>b*0.05: cv['Control Coverage'].append((c, max(0,min(1.0,(co-en)/(co-b)))))
    for c in rc:
        co=pa['projections_2050'].get(c,{}).get('Continued Trends',{}).get('mean_millions',100)
        dg=pa['projections_2050'].get(c,{}).get('Degraded Response',{}).get('mean_millions',co)
        if co>0: cv['Climate Impact'].append((c, min(1.0,(dg-co)/co)))
    return cv

out={'regions':regions,'metrics':metrics,'values':{},'country_values':{},'std':{},'n':{}}
print(f"{'Region':16}{'Metric':20}{'n':>3} {'mean':>7} {'SD':>7}  countries")
for region in regions:
    cv=pcv(region); out['country_values'][region]={}; out['values'][region]={}; out['std'][region]={}; out['n'][region]={}
    for m in metrics:
        vals=[v for _,v in cv[m]]; n=len(vals)
        out['country_values'][region][m]=[[c,round(v,4)] for c,v in cv[m]]
        out['n'][region][m]=n
        out['values'][region][m]=float(np.mean(vals)) if vals else existing['values'][region][m]
        out['std'][region][m]=float(np.std(vals,ddof=1)) if n>1 else 0.0
        cs=", ".join(f"{c}={v:.2f}" for c,v in cv[m])
        print(f"{region:16}{m:20}{n:>3} {out['values'][region][m]:>7.3f} {out['std'][region][m]:>7.3f}  {cs}")
    print()
json.dump(out, open('/tmp/panel_d_enhanced.json','w'), indent=1)
print("=== sanity: my mean vs existing paper bar (should be ~equal) ===")
for region in regions:
    for m in metrics:
        d=abs(out['values'][region][m]-existing['values'][region][m])
        flag='' if d<0.02 else '  <-- DIFFERS'
        if flag: print(f"  {region} / {m}: mine={out['values'][region][m]:.3f} vs paper={existing['values'][region][m]:.3f}{flag}")
print("  (only differences >0.02 shown)")
