import plotly.graph_objects as go
import pandas as pd

class MarketVisualizer:
    @staticmethod
    def plot_regimes(df: pd.DataFrame, symbol: str):
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol
        ))

        # Regime Lines
        # Identify groups of identical regimes to draw continuous lines
        df['regime_change'] = df['regime'].diff().fillna(0) != 0
        df['group'] = df['regime_change'].cumsum()

        for _, segment in df.groupby('group'):
            regime = segment['regime'].iloc[0]
            color = 'green' if regime == 0 else 'red'
            
            fig.add_trace(go.Scatter(
                x=segment.index, y=segment['close'],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            ))

        fig.update_layout(title=f"{symbol} Regimes", template="plotly_dark", xaxis_rangeslider_visible=False)
        return fig