
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class CurveBuilder:
    @staticmethod
    def _is_business_day(date):
        return date.weekday() < 5

    @staticmethod
    def _adjust_modified_following(date):
        original_month = date.month
        while not CurveBuilder._is_business_day(date):
            date += timedelta(days=1)
            if date.month != original_month:
                date -= timedelta(days=1)
                while not CurveBuilder._is_business_day(date):
                    date -= timedelta(days=1)
        return date

    @staticmethod
    def load_from_excel(file_path, value_date):
        """Load market rates from Excel file"""
        import pandas as pd

        df = pd.read_excel(file_path, header=1)  # Skip the "DISCOUNT CURVE" header
        market_rates = {
            row['Term']: float(row['Market Rate'])/100  # Convert percentage to decimal
            for _, row in df.iterrows()
        }
        return CurveBuilder(market_rates, value_date)

    def __init__(self, market_rates, value_date):
        self.market_rates = market_rates
        self.value_date = self._adjust_modified_following(value_date)
        self.short_term_cutoff = relativedelta(years=1)
        self.day_count = 360
        self.spline = None

    def get_tenor_date(self, tenor_str):
        try:
            parts = tenor_str.split()
            if len(parts) != 2:
                raise ValueError("Invalid tenor format. Expected format: '1 WK', '1 MO', or '1 YR'")

            value, unit = parts
            value = int(value)

            current_date = self.value_date
            if unit == 'WK':
                unadjusted_date = current_date + timedelta(weeks=value)
            elif unit == 'MO':
                unadjusted_date = current_date + relativedelta(months=value)
            elif unit == 'YR':
                unadjusted_date = current_date + relativedelta(years=value)
            else:
                raise ValueError("Invalid tenor unit. Must be WK, MO, or YR")

            return self._adjust_modified_following(unadjusted_date)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid tenor format: {tenor_str}") from e

    def is_short_term(self, tenor_str):
        tenor_date = self.get_tenor_date(tenor_str)
        cutoff_date = self.value_date + self.short_term_cutoff
        return tenor_date <= cutoff_date

    def calculate_dcf(self, start_date, end_date):
        return (end_date - start_date).days / self.day_count

    def short_term_df(self, rate, start_date, end_date):
        dcf = self.calculate_dcf(start_date, end_date)
        return 1 / (1 + rate * dcf)

    def long_term_df(self, rate, start_date, end_date, payment_frequency=4):
        dcf = self.calculate_dcf(start_date, end_date)
        periods = dcf * payment_frequency
        return (1 + rate/payment_frequency) ** (-periods)

    def bootstrap_zero_rates(self):
        tenor_dates = []
        zero_rates = []

        sorted_tenors = sorted(
            self.market_rates.items(),
            key=lambda x: self.get_tenor_date(x[0])
        )

        for tenor, rate in sorted_tenors:
            tenor_date = self.get_tenor_date(tenor)
            dcf = self.calculate_dcf(self.value_date, tenor_date)

            if self.is_short_term(tenor):
                df = self.short_term_df(rate, self.value_date, tenor_date)
                zero_rate = -np.log(df) / dcf
            else:
                zero_rate = np.log(1 + rate/4) * 4

            tenor_dates.append(tenor_date)
            zero_rates.append(zero_rate)

        return np.array(tenor_dates), np.array(zero_rates)

    def build_cubic_spline(self, tenor_dates, zero_rates):
        time_points = np.array([
            self.calculate_dcf(self.value_date, date)
            for date in tenor_dates
        ])
        return CubicSpline(time_points, zero_rates, bc_type='natural')

    def optimize_long_term_rates(self, initial_rates, tenor_dates):
        optimized_rates = initial_rates.copy()

        for tenor, market_rate in sorted(self.market_rates.items()):
            if not self.is_short_term(tenor):
                # For long-term rates, use simple conversion from market to zero rate
                tenor_date = self.get_tenor_date(tenor)
                dcf = self.calculate_dcf(self.value_date, tenor_date)
                rate_idx = len([t for t in tenor_dates if t <= tenor_date]) - 1
                optimized_rates[rate_idx] = np.log(1 + market_rate/4) * 4

        return optimized_rates

    def build_curve(self):
        tenor_dates, zero_rates = self.bootstrap_zero_rates()
        optimized_rates = self.optimize_long_term_rates(zero_rates, tenor_dates)
        self.spline = self.build_cubic_spline(tenor_dates, optimized_rates)
        return self.spline, tenor_dates, optimized_rates

    def get_discount_factor(self, date):
        if date < self.value_date:
            raise ValueError("Date cannot be before value date")

        t = self.calculate_dcf(self.value_date, date)
        zero_rate = self.spline(t)
        return np.exp(-zero_rate * t)

    def get_forward_rate(self, start_date, end_date):
        if start_date < self.value_date:
            raise ValueError("Start date cannot be before value date")
        if end_date <= start_date:
            raise ValueError("End date must be after start date")

        df1 = self.get_discount_factor(start_date)
        df2 = self.get_discount_factor(end_date)
        dcf = self.calculate_dcf(start_date, end_date)

        return (df1/df2 - 1) / dcf

    def print_curve_data(self, file='curve_output.xlsx'):
        """Print curve data to Excel file and console"""
        import pandas as pd

        data = []
        for tenor, market_rate in self.market_rates.items():
            tenor_date = self.get_tenor_date(tenor)
            dcf = self.calculate_dcf(self.value_date, tenor_date)
            zero_rate = self.spline(dcf)
            df = self.get_discount_factor(tenor_date)

            data.append({
                'Term': tenor,
                'Market Rate': market_rate * 100,
                'Shift': 0,
                'Shifted Rate': market_rate * 100,
                'Zero Rate': zero_rate * 100,
                'Discount': df
            })

        df = pd.DataFrame(data)
        df.to_excel(file, index=False, sheet_name='DISCOUNT CURVE')
        print("\nCurve Data:")
        print(df.to_string())
        return df

if __name__ == '__main__':
    from datetime import datetime

    value_date = datetime(2025, 1, 14)
    curve = CurveBuilder.load_from_excel('./tests/ois_input.xlsx', value_date)
    curve.build_curve()
    curve.print_curve_data()
