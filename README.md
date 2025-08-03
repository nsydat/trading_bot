# 🤖 Trading Bot Tự Động - Binance Futures

Bot trading tự động 24/7 với Machine Learning và quản lý rủi ro nâng cao.

## ✨ Tính Năng Chính

- 🔄 Giao dịch tự động 24/7
- 🧠 Machine Learning tích hợp
- 📊 Multi-timeframe analysis  
- ⚡ Multi-strategy support
- 🛡️ Risk management nghiêm ngặt
- 📱 Thông báo real-time
- 📈 Dashboard giám sát
- 🐳 Docker deployment ready

## 🚀 Quick Start

### 1. Cài Đặt Dependencies
```bash
pip install -r deployment/requirements.txt
```

### 2. Cấu Hình
```bash
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn
```

### 3. Chạy Testnet
```bash
python main.py --testnet
```

### 4. Chạy Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

## 📁 Cấu Trúc Dự Án

Xem chi tiết trong docs/architecture/

## 🛠️ Development

### Testing
```bash
pytest tests/
```

### Backtesting
```bash
python scripts/backtest_strategy.py --strategy ema_crossover
```

## 📧 Support

Mọi thắc mắc vui lòng tạo issue trên GitHub.
