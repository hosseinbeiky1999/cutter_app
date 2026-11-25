import streamlit as st
import pandas as pd
import pulp
from collections import defaultdict
import numpy as np

# ================== تنظیمات و قوانین ==================
EPS = 1e-9

PROFILES = {
    "چارچوب دربی":   [(5, 78), (98, 160)],
    "لنگه دربی":     [(5, 78), (98, 160)],
    "چارچوب پنجره":  [(5, 78), (90, 110)],
    "لنگه پنجره":    [(5, 57), (66, 118)],
    "سپری":          [(5, 60)],
    "زوار":          [(7, 50)],
}

def is_allowed_remainder(profile, r):
    if r < -EPS:
        return False
    if abs(r) < EPS:
        return True
    for lo, hi in PROFILES[profile]:
        if lo - EPS <= r <= hi + EPS:
            return False
    return True

# ================== الگوریتم برش ==================
def generate_patterns(bar_length, lengths, qty, kerf, is_allowed):
    patterns = []
    n = len(lengths)
    max_counts = []
    
    for L, Q in zip(lengths, qty):
        m = int(bar_length // (L + kerf)) if L + kerf > 0 else 0
        max_counts.append(min(Q, m))

    def rec(idx, counts, used, pieces):
        if used + pieces * kerf > bar_length + EPS:
            return
        if idx == n:
            if pieces == 0:
                return
            rem = bar_length - used - pieces * kerf
            if rem < -EPS:
                return
            if abs(rem) < EPS:
                rem = 0.0
            if is_allowed(rem):
                patterns.append((tuple(counts), used, pieces, rem))
            return
        L = lengths[idx]
        for c in range(max_counts[idx] + 1):
            rec(idx + 1, counts + [c], used + c * L, pieces + c)

    rec(0, [], 0, 0)
    return patterns

def solve_cutting(profile, pieces, stock_bars, kerf):
    demand = defaultdict(float)
    for L, Q in pieces:
        if Q > 0 and L > 0:
            demand[L] += Q
    if not demand:
        return None, "هیچ قطعه معتبری وجود ندارد."

    lengths = list(demand.keys())
    qty = [int(round(demand[L])) for L in lengths]
    
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    lengths = [lengths[i] for i in order]
    qty = [qty[i] for i in order]

    allowed_fn = lambda r: is_allowed_remainder(profile, r)
    all_patterns = [] 
    
    for stock_idx, (bar_len, bar_qty) in enumerate(stock_bars):
        if bar_qty > 0 and bar_len > 0:
            pats = generate_patterns(bar_len, lengths, qty, kerf, allowed_fn)
            for p in pats:
                all_patterns.append((stock_idx, p[0], p[1], p[2], p[3]))

    if not all_patterns:
        return None, "هیچ ترکیبی از برش پیدا نشد که ته شاخه در منطقه ممنوعه این پروفیل نیفتد."

    solver = pulp.COIN_CMD(msg=False)
    prob = pulp.LpProblem("cutting_stock", pulp.LpMinimize)
    
    x = [prob.add_variable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(all_patterns))]

    prob += pulp.lpSum(all_patterns[i][4] * x[i] for i in range(len(all_patterns)))

    for j, q in enumerate(qty):
        prob += pulp.lpSum(all_patterns[i][1][j] * x[i] for i in range(len(all_patterns))) == q

    for stock_idx, (bar_len, bar_qty) in enumerate(stock_bars):
        if bar_qty > 0:
            prob += pulp.lpSum(x[i] for i in range(len(all_patterns)) if all_patterns[i][0] == stock_idx) <= bar_qty

    status = prob.solve(solver)
    if status.status_str != "Optimal":
        return None, "با این موجودی و محدودیت‌ها، امکان برش وجود ندارد."

    rows = []
    for i, var in enumerate(x):
        k = int(round(var.varValue or 0))
        if k > 0:
            stock_idx, counts, used, pn, rem = all_patterns[i]
            bar_len = stock_bars[stock_idx][0]
            parts = [(lengths[j], counts[j]) for j in range(len(lengths)) if counts[j] > 0]
            rows.append({
                "bar_len": bar_len,
                "parts": parts, 
                "n": pn, 
                "used": used, 
                "kerf": pn * kerf, 
                "rem": rem,
                "qty_used": k
            })

    rows.sort(key=lambda r: (r["bar_len"], r["rem"]))
    return rows, None
# ================== رابط گرافیکی وب (Streamlit) ==================
st.set_page_config(page_title="برش پروفیل", layout="wide")
st.title("🔪 برنامه‌ریزی بهینه برش پروفیل")

st.subheader("⚙️ تنظیمات")
col1, col2 = st.columns(2)
with col1:
    kerf_mm = st.number_input("کلفتی تیغ (میلی‌متر):", min_value=0.0, value=4.0, step=0.5)
kerf_cm = kerf_mm / 10.0

with col2:
    profile = st.selectbox("نوع پروفیل:", list(PROFILES.keys()))

ranges_txt = "  |  ".join(f"{lo} تا {hi} ممنوع" for lo, hi in PROFILES[profile])
st.info(f"محدوده‌های ممنوعه ته شاخه: {ranges_txt}")

st.write("---")

st.subheader("📦 موجودی شاخه‌ها در انبار")
st.caption("اگر تعداد یک شاخه نامحدود است، عدد بزرگی مثل 99999 وارد کنید. طول‌ها به سانتی‌متر هستند (۶ متر = 600).")

if "stock_df" not in st.session_state:
    st.session_state.stock_df = pd.DataFrame([{"طول شاخه (cm)": 600.0, "تعداد موجود": 99999}])

stock_df = st.data_editor(
    st.session_state.stock_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "طول شاخه (cm)": st.column_config.NumberColumn(min_value=1.0, step=10.0),
        "تعداد موجود": st.column_config.NumberColumn(min_value=1, step=1)
    }
)

st.write("---")

st.subheader("📝 ابعاد قطعات مورد نیاز")
st.caption("طول‌ها به سانتی‌متر (مثلاً 224.1) وارد شوند. حتماً هم تعداد و هم ابعاد را پر کنید.")

if "pieces_df" not in st.session_state:
    st.session_state.pieces_df = pd.DataFrame([{"تعداد": 1, "ابعاد (cm)": None}])

pieces_df = st.data_editor(
    st.session_state.pieces_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "تعداد": st.column_config.NumberColumn(min_value=1, step=1),
        "ابعاد (cm)": st.column_config.NumberColumn(min_value=1.0, step=0.1)
    }
)

st.write("---")

if st.button("🚀 محاسبه نقشه برش", type="primary", use_container_width=True):
    
    pieces = []
    for _, row in pieces_df.iterrows():
        q = row["تعداد"]
        l = row["ابعاد (cm)"]
        # بررسی اینکه آیا ابعاد خالی است یا نه
        if pd.isna(q) or pd.isna(l):
            continue
        q = int(q)
        l = float(l)
        if q > 0 and l > 0:
            pieces.append((l, q))
            
    stock_bars = []
    for _, row in stock_df.iterrows():
        bl = row["طول شاخه (cm)"]
        bq = row["تعداد موجود"]
        if pd.isna(bl) or pd.isna(bq):
            continue
        bl = float(bl)
        bq = int(bq)
        if bl > 0 and bq > 0:
            stock_bars.append((bl, bq))

    # اگر ورودی‌ها ناقص بودند، اینجا متوقف شو تا برنامه کرش نکند
    if not pieces:
        st.error("❌ هیچ قطعه معتبری وارد نشده است. لطفاً هم «تعداد» و هم «ابعاد» را در جدول پر کنید.")
        st.stop()
        
    if not stock_bars:
        st.error("❌ هیچ شاخه معتبری در انبار تعریف نشده است. لطفاً جدول انبار را بررسی کنید.")
        st.stop()

    with st.spinner("در حال محاسبه بهینه..."):
        rows, err = solve_cutting(profile, pieces, stock_bars, kerf_cm)
        
        if err:
            st.error(f"❌ {err}")
            st.stop()
            
        st.success("✅ نقشه برش با موفقیت محاسبه شد.")
        
        table_data = []
        total_waste = 0
        total_bars = 0
        
        # این حلقه فقط زمانی اجرا می‌شود که rows وجود داشته باشد
        for r in rows:
            parts_str = " + ".join(f"{L:g}×{c}" for L, c in r["parts"])
            table_data.append({
                "طول شاخه (cm)": f"{r['bar_len']:.0f}",
                "تعداد شاخه مصرفی": r["qty_used"],
                "قطعات بریده شده": parts_str,
                "تعداد قطعات": r["n"],
                "جمع طول قطعات": f"{r['used']:.2f}",
                "تیغ": f"{r['kerf']:.2f}",
                "ته شاخه (ضایعات)": f"{r['rem']:.2f}"
            })
            total_waste += r["rem"] * r["qty_used"]
            total_bars += r["qty_used"]
        
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        st.info(f"تعداد کل شاخه‌های مصرفی: {total_bars} عدد  |  مجموع ضایعات: {total_waste:.2f} سانتی‌متر ({total_waste/100:.3f} متر)")
