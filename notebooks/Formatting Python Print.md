# Python Print Formatting  

When you combine the **sign options** (`+`, `-`, or space) with **width, precision, and alignment**, you can make your numbers line up beautifully in tables, reports, or chart labels.  

Here’s a quick cheat sheet:  

| Format | Meaning | Example (`x = 5.2`, `y = -5.2`) |
|--------|---------|---------------------------------|
| `+8.2f` | Always show sign, width 8, 2 decimals | `+5.20` / `-5.20` |
| `-8.2f` | Default sign rule, width 8, 2 decimals | `5.20` / `-5.20` |
| `8.2f` | Space for positive, width 8, 2 decimals | `␣5.20` / `-5.20` |
| `<+8.2f` | Left-align, always show sign | `+5.20` / `-5.20` |
| `^+8.2f` | Center-align, always show sign | ` +5.20 ` / ` -5.20 ` |

Example in code:  

```python
x, y = 5.2, -5.2
for fmt in ["+8.2f", "-8.2f", " 8.2f", "<+8.2f", "^+8.2f"]:
    print(f"{x:{fmt}} | {y:{fmt}}")
```

This way, whether you’re printing a **P&L table**, **Sharpe ratios**, or **model metrics**, everything lines up neatly and is easy to read.  

If you want, I can show you how to apply this to **financial backtest results** so your metrics look like a professional tear sheet. Would you like me to do that next?
