# Signal Tracker Class - User Guide

You have created BetSizing.mqh (which holds the *calculations*) and SignalTracker.mqh (which holds the *state*). This guide explains how to use them together in your Expert Advisor (EA).

The CSignalTracker class is the "brain" of your EA. You just need to tell it when your signals start and stop, and it will handle all the complex state tracking for you.

## 1. Setup in Your EA

At the top of your EA's .mq5 file, include the new tracker class. **Make sure BetSizing.mqh is in the same folder or in your main MQL5/Include folder.**

```cpp
//+------------------------------------------------------------------+  
//|                                                   MyStrategy.mq5 |  
//+------------------------------------------------------------------+  
#include <Trade\Trade.mqh>  
#include "SignalTracker.mqh" // Include the new class

// --- Global Variables ---  
CTrade            trade;  
CSignalTracker    g_tracker; // Create one global instance of the tracker

// --- EA Inputs ---  
input double  MaxPositionSize = 10.0; // Max lots  
//... your other inputs
```

## 2. How to Use the CSignalTracker

Your EA's main job is to tell the g_tracker object what is happening.

### OnSignalNew(int side)

Call this **once** when your model generates a new signal.

### OnSignalClose(int side)

Call this **once** when that same signal's *event* ends (e.g., its t1 is reached, a position is closed, or an opposite signal arrives).

### SetCurrentPosition(double pos)

Call this to update the tracker with your *actual* net position. This is crucial for bet_size_dynamic functions.

## 3. Usage Examples

Here is how you would use the tracker to implement each bet sizing strategy.

### Example 1: BetSizeBudget

This is the simplest. The tracker handles everything.

```cpp
void OnNewSignalDetected(int side) // side=1 or -1  
{  
   // 1. Tell the tracker  
   g_tracker.OnSignalNew(side);  
     
   // 2. Get the new bet size  
   double bet_size = g_tracker.GetBetSizeBudget();  
     
   // 3. Calculate volume and trade  
   double volume = bet_size * MaxPositionSize;  
   // ... execute trade ...  
}

void OnSignalExpired(int side) // side=1 or -1  
{  
   // 1. Tell the tracker  
   g_tracker.OnSignalClose(side);  
}
```

### Example 2: BetSizeReserve

This is just as simple from the EA's perspective. The tracker manages the c_t history internally.

```cpp
void OnNewSignalDetected(int side) // side=1 or -1  
{  
   // 1. Tell the tracker  
   g_tracker.OnSignalNew(side);  
     
   // 2. Get the new bet size  
   // This will return 0.0 until it has enough history  
   double bet_size = g_tracker.GetBetSizeReserve();  
     
   if(g_tracker.GetCtHistoryTotal() \> 20\) // Wait for some history  
   {  
      // 3. Calculate volume and trade  
      double volume = bet_size * MaxPositionSize;  
      // ... execute trade ...  
   }  
}

// Remember to call g_tracker.OnSignalClose(side) when it expires!
```

### Example 3: BetSizeDynamic (Sigmoid)

This workflow is more involved, as it combines the tracker's *state* with *parameters* you provide.

**Step A: Calibration (Do this once, e.g., in OnInit)**
You need to find your w_param. Let's say you calibrate that a price divergence of 0.00500 (500 points) should result in a 0.95 bet size.  
double g_w_param = 0.0;

```cpp
int OnInit()  
{  
   // ...  
   double cal_divergence = 0.00500; // e.g., 500 points  
   double cal_bet_size = 0.95;  
     
   g_w_param = GetWSigmoid(cal_divergence, cal_bet_size);  
   Print("w_param calibrated to: ", g_w_param);  
   // ...  
   return(INIT_SUCCEEDED);  
}
```

**Step B: Real-Time Usage (e.g., in OnTick)**  
In OnTick, you get new prices and model forecasts.  

```cpp
void OnTick()  
{  
   // 1. Get real-time data  
   double market_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);  
   double forecast_price = GetMyModelForecast(); // Your magic function  
     
   // 2. Update tracker's state  
   // (You need logic to get your *actual* net position)  
   double current_net_pos = GetCurrentNetPosition();   
   g_tracker.SetCurrentPosition(current_net_pos);

   // 3. Get the Target Position from BetSizing.mqh  
   double target_pos = GetTargetPosSigmoid(  
      g_w_param,  
      forecast_price,  
      market_price,  
      MaxPositionSize  
   );  
     
   // 4. Get the current position from the tracker  
   double current_pos = g_tracker.GetCurrentPosition();  
     
   // 5. Decide to trade  
   if(target_pos != current_pos)  
   {  
      // 6. Get the Limit Price from BetSizing.mqh  
      double limit_price = LimitPriceSigmoid(  
         target_pos,  
         current_pos,  
         forecast_price,  
         g_w_param,  
         MaxPositionSize  
      );  
        
      double volume_to_trade = target_pos - current_pos;  
        
      Print("New trade! Target: ", target_pos, ", Current: ", current_pos);  
      Print("Placing order for ", volume_to_trade, " lots at limit ", limit_price);  
        
      // ... execute limit order ...  
        
      // NOTE: You do NOT call OnSignalNew/Close here.  
      // This system is position-based, not signal-event-based.  
      // You just update the current position state.  
   }  
}  
```
