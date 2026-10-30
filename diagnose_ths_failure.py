import json

# Check why Tonghuashun formula fails to select
# 1. Check if 000678 had ZT1 and ZT2 on 10-27 and 10-28:
# On 10-28:
# Yesterday = 10-27 (ZT)
# Day before = 10-26 (ZT)
# Gap on 10-28 = +5.93% (in [4.5, 8.5])
# Amount on 10-27 = 14.92亿 (>= 3.5亿)
# Turnover on 10-27 = 20.6% (in [12, 26])

# Why did THS return 0?
# Potential issues in THS syntax:
# 1. In THS, REF(VOL,1) / CAPITAL * 100:
# If CAPITAL in THS is 0 or null during offline selection?
# 2. In THS, NAMELIKE('ST') = 0:
# If NAMELIKE returns 0 when true or 1 when true?
# In THS, NAMELIKE('ST') returns 1 if name contains ST!
# So NAMELIKE('ST') = 0 means NOT ST. That is correct.
# BUT what if CODELIKE('30') = 0?
# Does CODELIKE exist in THS 9.60?
# In THS 9.60 (the user's version):
# If CODELIKE is NOT supported or behaves differently, MAINBOARD becomes 0!
# If MAINBOARD is 0, XG is ALWAYS 0!

print("Analysis of potential formula blockers in Tonghuashun PC 9.60:")
