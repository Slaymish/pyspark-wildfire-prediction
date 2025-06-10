#!/usr/bin/env python3
"""
Preprocessing script for US Wildfire dataset
Converts SQLite database to CSV format suitable for Spark processing
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

def load_data_from_sqlite(db_path):
    """Load data from SQLite database"""
    print("Loading data from SQLite database...")
    
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            FOD_ID, FPA_ID, SOURCE_SYSTEM_TYPE, SOURCE_SYSTEM,
            NWCG_REPORTING_AGENCY, NWCG_REPORTING_UNIT_ID, NWCG_REPORTING_UNIT_NAME,
            SOURCE_REPORTING_UNIT, SOURCE_REPORTING_UNIT_NAME,
            LOCAL_FIRE_REPORT_ID, LOCAL_INCIDENT_ID, FIRE_CODE, FIRE_NAME,
            ICS_209_INCIDENT_NUMBER, ICS_209_NAME, MTBS_ID, MTBS_FIRE_NAME,
            COMPLEX_NAME, FIRE_YEAR, DISCOVERY_DATE, DISCOVERY_DOY, DISCOVERY_TIME,
            STAT_CAUSE_CODE, STAT_CAUSE_DESCR, CONT_DATE, CONT_DOY, CONT_TIME,
            FIRE_SIZE, FIRE_SIZE_CLASS, LATITUDE, LONGITUDE,
            OWNER_CODE, OWNER_DESCR, STATE, COUNTY, FIPS_CODE, FIPS_NAME
        FROM Fires
        WHERE FIRE_SIZE IS NOT NULL 
        AND LATITUDE IS NOT NULL 
        AND LONGITUDE IS NOT NULL
        AND FIRE_SIZE_CLASS IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Loaded {len(df)} records from database")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def engineer_temporal_features(df):
    """Create temporal features from dates"""
    print("Engineering temporal features...")
    df['DISCOVERY_DATE'] = pd.to_datetime(df['DISCOVERY_DATE'], errors='coerce')
    df['CONT_DATE'] = pd.to_datetime(df['CONT_DATE'], errors='coerce')
    df['discovery_month'] = df['DISCOVERY_DATE'].dt.month
    df['discovery_year'] = df['FIRE_YEAR']
    df['discovery_day_of_week'] = df['DISCOVERY_DATE'].dt.dayofweek
    df['discovery_day_of_year'] = df['DISCOVERY_DOY']
    df['season'] = df['discovery_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    df['fire_duration_days'] = (df['CONT_DATE'] - df['DISCOVERY_DATE']).dt.days
    df['fire_duration_days'] = df['fire_duration_days'].fillna(0)
    df['decade'] = (df['FIRE_YEAR'] // 10) * 10
    df['fire_season'] = ((df['discovery_month'] >= 5) & (df['discovery_month'] <= 10)).astype(int)
    df['is_weekend'] = (df['discovery_day_of_week'] >= 5).astype(int)
    df['DISCOVERY_TIME'] = pd.to_numeric(df['DISCOVERY_TIME'], errors='coerce').fillna(1200)
    df['discovery_hour'] = (df['DISCOVERY_TIME'] // 100) % 24
    return df

def engineer_spatial_features(df):
    """Create spatial features"""
    print("Engineering spatial features...")
    df['lat_bin'] = pd.cut(df['LATITUDE'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['LONGITUDE'], bins=20, labels=False)
    def get_region(lat, lon):
        if lat >= 47 and lon <= -120: return 'PNW'
        elif lat >= 37 and lon <= -120: return 'CA'
        elif lat >= 37 and lon <= -100: return 'MW'
        elif lat >= 37: return 'North'
        elif lon <= -100: return 'SW'
        else: return 'SE'
    df['geographic_region'] = df.apply(lambda x: get_region(x['LATITUDE'], x['LONGITUDE']), axis=1)
    return df

def engineer_categorical_features(df):
    """Process categorical features"""
    print("Engineering categorical features...")
    categorical_columns = [
        'SOURCE_SYSTEM_TYPE', 'NWCG_REPORTING_AGENCY', 'STAT_CAUSE_DESCR',
        'OWNER_DESCR', 'STATE', 'COUNTY', 'season', 'geographic_region'
    ]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    df['state_fire_frequency'] = df['STATE'].map(df['STATE'].value_counts())
    df['county_fire_frequency'] = df['COUNTY'].map(df['COUNTY'].value_counts())
    df['cause_frequency'] = df['STAT_CAUSE_DESCR'].map(df['STAT_CAUSE_DESCR'].value_counts())
    df['agency_fire_frequency'] = df['NWCG_REPORTING_AGENCY'].map(df['NWCG_REPORTING_AGENCY'].value_counts())
    return df

def engineer_fire_features(df):
    """Create fire-specific features"""
    print("Engineering fire-specific features...")
    

    ### REMOVED CUS DATA LEAKAGE #####

    # The following features are derived from FIRE_SIZE, which is the target.
    # df['fire_size_log'] = np.log1p(df['FIRE_SIZE'])
    # df['fire_size_class_numeric'] = df['FIRE_SIZE_CLASS'].map(size_class_map)
    # df['large_fire'] = (df['FIRE_SIZE'] > 1000).astype(int)
    # df['fire_size_state_percentile'] = df.groupby('STATE')['FIRE_SIZE'].rank(pct=True)
    
    # This feature is safe as it's based on cause, not size.
    human_causes = ['Arson', 'Campfire', 'Children', 'Debris Burning', 'Equipment Use',
                   'Fireworks', 'Powerline', 'Railroad', 'Smoking', 'Structure']
    df['human_caused'] = df['STAT_CAUSE_DESCR'].isin(human_causes).astype(int)
    return df

def create_risk_indicators(df):
    """Create risk-based features"""
    print("Creating risk indicators...")
    high_risk_months = [6, 7, 8, 9]
    df['high_risk_month'] = df['discovery_month'].isin(high_risk_months).astype(int)
    drought_months = [7, 8, 9, 10]
    df['drought_season'] = df['discovery_month'].isin(drought_months).astype(int)
    location_counts = df.groupby(['lat_bin', 'lon_bin']).size()
    df['location_fire_density'] = df.apply(
        lambda x: location_counts.get((x['lat_bin'], x['lon_bin']), 0), axis=1
    )
    return df

def select_final_features(df):
    """Select final feature set for modeling, excluding leaking variables."""
    print("Selecting final features...")
    
    final_features = [
        # Target variable
        'FIRE_SIZE_CLASS',
        
        # Identifiers
        'FOD_ID', 'STATE', 'COUNTY',
        
        # Temporal features
        'discovery_month', 'discovery_year', 'discovery_day_of_week', 
        'discovery_day_of_year', 'season', 'decade', 'fire_duration_days',
        'discovery_hour',
        
        # Spatial features
        'LATITUDE', 'LONGITUDE', 'lat_bin', 'lon_bin', 
        'geographic_region', 'location_fire_density',
        
        # Fire characteristics (non-leaking)
        'human_caused', 'STAT_CAUSE_CODE', 'STAT_CAUSE_DESCR',
        
        # Administrative features
        'SOURCE_SYSTEM_TYPE', 'NWCG_REPORTING_AGENCY', 'OWNER_DESCR',
        'state_fire_frequency', 'county_fire_frequency', 'cause_frequency',
        'agency_fire_frequency', 'FIPS_CODE', 'OWNER_CODE', 'SOURCE_SYSTEM',
        
        # Risk indicators
        'fire_season', 'is_weekend', 'high_risk_month', 'drought_season',
    ]
    
    available_features = list(set([f for f in final_features if f in df.columns]))
    df_final = df[available_features].copy()
    
    print(f"Selected {len(available_features)} non-leaking features for modeling")
    return df_final


def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing...")

    db_path = "FPA_FOD_20170508.sqlite"
    output_path = "data/wildfire_processed_no_leakage.csv"
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        print("Please run the download_and_process.sh script first.")
        sys.exit(1)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data_from_sqlite(db_path)
    df = engineer_temporal_features(df)
    df = engineer_spatial_features(df)
    df = engineer_categorical_features(df)
    df = engineer_fire_features(df)
    df = create_risk_indicators(df)
    df_final = select_final_features(df)
    
    df_final = df_final.dropna(subset=['FIRE_SIZE_CLASS'])
    
    df_final.to_csv(output_path, index=False)
    
    print(f"Preprocessing complete! Final dataset shape: {df_final.shape}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
