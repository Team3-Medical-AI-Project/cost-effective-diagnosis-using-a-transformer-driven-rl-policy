# Cost Model for Sepsis Diagnostic Panels

The following costs are used as penalties within the Reinforcement Learning environment's reward function. The prices are based on national average cash prices from direct-to-consumer laboratories in the United States, serving as a realistic proxy for the financial cost of each test panel.

| Action Index | Panel Name | Final USD Cost | Source / Justification |
| 0 | **Complete Blood Count (CBC)** | **$30.86** | National average of state-level cash prices. |
| 1 | **Comprehensive Metabolic Panel (CMP)** | **$67.73** | National average of state-level cash prices. |
| 2 | **Arterial Blood Gas (ABG)** | **$509.00**| Estimated US national average cash price. |
| 3 | **aPTT / Coagulation Panel** | **$49.00** | Average of major national lab (Labcorp, Quest) cash prices. |