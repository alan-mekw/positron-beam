#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import scipy.optimize as scpo
import scipy.stats as sps
import matplotlib.pyplot as plt


E   = np.array([511, 1274.53, 121.782, 244.697, 344.279, 778.904, 964.079, 1408.006])
Ch  = np.array([2228.19, 5777.13, 413.23, 984.56, 1447.41, 3468.25, 4329.12, 6393.4])
Ch_er = np.full_like(Ch, 0.05)          # REPLACE with real fit uncertainties
Fw_keV = np.array([2.91, 1.2, 3.47, 3.63, 3.71, 3.89, 4.09, 4.26])
Fw_er  = np.full_like(Fw_keV, 0.05)     # REPLACE with real fit uncertainties

def line(x, m, c):
    return m * x + c

def chi2(y, yfit, sig):
    return np.sum(((y - yfit) / sig) ** 2)

def report(name, popt, cov, y, yfit, sig):
    c2 = chi2(y, yfit, sig); dof = len(y) - len(popt)
    print(f"{name}: chi2 = {c2:.2f}, dof = {dof}, reduced = {c2/dof:.2f}, p = {sps.chi2.sf(c2, dof):.3g}")
    for n, v, e in zip(["slope", "intercept"], popt, np.sqrt(np.diag(cov))):
        print(f"   {n} = {v:.5f} +/- {e:.5f}")



# In[26]:


# Calibration: channel = m*E + c
p0 = np.polyfit(E, Ch, 1)               # good starting values
popt, cov = scpo.curve_fit(line, E, Ch, p0=p0, sigma=Ch_er, absolute_sigma=True)
report("Calibration (ch vs E)", popt, cov, Ch, line(E, *popt), Ch_er)
m, c = popt
resid = Ch - line(E, *popt)
print("Residuals (ch):", np.round(resid, 2))

fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
xs = np.linspace(E.min(), E.max(), 200)
ax[0].errorbar(E, Ch, yerr=Ch_er, fmt='o'); ax[0].plot(xs, line(xs, *popt), 'r')
ax[0].set_ylabel('Channel')
ax[1].errorbar(E, resid, yerr=Ch_er, fmt='o'); ax[1].axhline(0, color='r')
ax[1].set_xlabel('Line energy (keV)'); ax[1].set_ylabel('Residual (ch)')
plt.tight_layout(); plt.savefig('calibration.png', dpi=150)


# In[27]:


# Resolution: FWHM^2 = a + b*E  (FWHM in keV)
y = Fw_keV ** 2
y_er = 2 * Fw_keV * Fw_er               # error propagation
p0r = np.polyfit(E, y, 1)
poptr, covr = scpo.curve_fit(line, E, y, p0=p0r, sigma=y_er, absolute_sigma=True)
report("Resolution (FWHM^2 vs E)", poptr, covr, y, line(E, *poptr), y_er)
b, a = poptr                             # slope b, intercept a
print(f"a (noise term) = {a:.4f} keV^2, b = {b:.5f} keV, Fano F = {b/(8*np.log(2)*0.00296):.3f}")

plt.figure(figsize=(5, 3))
plt.errorbar(E, y, yerr=y_er, fmt='o'); plt.plot(xs, line(xs, *poptr), 'r')
plt.xlabel('Line energy (keV)'); plt.ylabel(r'FWHM$^2$ (keV$^2$)')
plt.tight_layout(); plt.savefig('resolution.png', dpi=150)


# In[29]:


xs = np.linspace(E.min(), E.max(), 200)
R_fit = 100*np.sqrt(np.maximum(a + b*xs, 0))/xs

R = Fw_keV/E*100
R_er = Fw_er/E*100

plt.figure(figsize=(5, 3))
plt.errorbar(E, R, yerr=R_er, fmt='o')
plt.plot(xs, R_fit, 'r')
plt.xlabel('Line energy (keV)'); plt.ylabel('FWHM/E (%)')
plt.tight_layout(); plt.savefig('resolution_R.png', dpi=150)


# In[ ]:




