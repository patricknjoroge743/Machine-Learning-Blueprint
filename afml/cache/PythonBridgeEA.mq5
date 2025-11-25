//+------------------------------------------------------------------+
//|                                               PythonBridgeEA.mq5 |
//|                                    MQL5 Client for AFML Bridge   |
//+------------------------------------------------------------------+
#property copyright "AFML Integration"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <JsonLib.mqh>
#include "..\..\Shared Projects\MQL5Book\Include\PRTF.mqh"

//--- Input parameters
input string   PythonHost = "127.0.0.1";            // Python server host (NO http://)
input int      PythonPort = 80;                     // Python server port
input double   RiskPercent = 1.0;                   // Risk per trade (%)
input int      MagicNumber = 12345;                 // EA magic number
input bool     EnableTrading = true;                // Enable actual trading
input int      HeartbeatInterval = 5000;            // Heartbeat interval (ms)
input int      DataSendInterval = 1000;             // Market data send interval (ms)

//--- Global variables
int socket_handle = INVALID_HANDLE;
CTrade trade;
datetime last_heartbeat = 0;
datetime last_data_send = 0;
bool is_connected = false;

//--- Message buffer
uchar message_buffer[];
int buffer_size = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
// Setup trade object
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

// Initialize buffer
   ArrayResize(message_buffer, 0);

// Connect to Python bridge
   if(!ConnectToPython())
     {
      Print("Failed to connect to Python bridge - will retry");
     }

   Print("Python Bridge EA initialized successfully");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   DisconnectFromPython();
   Print("Python Bridge EA deinitialized");
  }

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
  {
   static datetime last_reconnect = 0;

// Check connection
   if(!is_connected)
     {
      // Try reconnect every 5 seconds
      if(TimeCurrent() - last_reconnect >= 5)
        {
         Print("Attempting to reconnect to Python...");
         ConnectToPython();
         last_reconnect = TimeCurrent();
        }
      return;
     }

// Send heartbeat
   if(TimeCurrent() - last_heartbeat >= HeartbeatInterval / 1000)
     {
      SendHeartbeat();
      last_heartbeat = TimeCurrent();
     }

// Send market data
   if(TimeCurrent() - last_data_send >= DataSendInterval / 1000)
     {
      SendMarketData();
      last_data_send = TimeCurrent();
     }

// Check for incoming signals
   ProcessIncomingMessages();
  }

//+------------------------------------------------------------------+
//| Connect to Python bridge server                                 |
//+------------------------------------------------------------------+
bool ConnectToPython()
  {
   if(socket_handle != INVALID_HANDLE)
     {
      PRTF(SocketClose(socket_handle));
     }

// Create socket
   socket_handle = PRTF(SocketCreate());
   if(socket_handle == INVALID_HANDLE)
     {
      return false;
     }

// Connect to server
   if(!PRTF(SocketConnect(socket_handle, PythonHost, PythonPort, 5000)))
     {
      Print("Failed to connect to Python server");
      PRTF(SocketClose(socket_handle));
      socket_handle = INVALID_HANDLE;
      return false;
     }

   is_connected = true;
   Print("Connected to Python bridge at ", PythonHost, ":", PythonPort);
   Print("Socket handle: ", socket_handle);
   Print("Testing initial communication...");

// Test the connection immediately
   SendHeartbeat();

// Request any pending signals
   RequestPendingSignals();

   return true;
  }

//+------------------------------------------------------------------+
//| Disconnect from Python bridge                                   |
//+------------------------------------------------------------------+
void DisconnectFromPython()
  {
   if(socket_handle != INVALID_HANDLE)
     {
      PRTF(SocketClose(socket_handle));
      socket_handle = INVALID_HANDLE;
     }
   is_connected = false;
  }

//+------------------------------------------------------------------+
//| Send JSON message to Python with length prefix                  |
//+------------------------------------------------------------------+
bool SendMessage(MQL5_Json::JsonDocument &doc)
  {
   if(socket_handle == INVALID_HANDLE || !is_connected)
     {
      return false;
     }

// Serialize JSON to string (compact format)
   string json_message = doc.ToString(false);

// Convert string to UTF-8 byte array
   uchar data[];
   int str_len = StringToCharArray(json_message, data, 0, WHOLE_ARRAY, CP_UTF8);

// Remove null terminator if present
   if(str_len > 0 && data[str_len - 1] == 0)
      str_len--;

   ArrayResize(data, str_len);

// Create length prefix (4 bytes, little-endian)
   uint length = str_len;
   uchar length_bytes[4];
   length_bytes[0] = (uchar)(length & 0xFF);
   length_bytes[1] = (uchar)((length >> 8) & 0xFF);
   length_bytes[2] = (uchar)((length >> 16) & 0xFF);
   length_bytes[3] = (uchar)((length >> 24) & 0xFF);

// Send length prefix
   if(PRTF(SocketSend(socket_handle, length_bytes, 4)) != 4)
     {
      is_connected = false;
      return false;
     }

// Send data
   if(PRTF(SocketSend(socket_handle, data, str_len)) != str_len)
     {
      is_connected = false;
      return false;
     }

   return true;
  }

//+------------------------------------------------------------------+
//| Send heartbeat to Python                                        |
//+------------------------------------------------------------------+
void SendHeartbeat()
  {
   MQL5_Json::JsonDocument doc = MQL5_Json::JsonNewObject();
   MQL5_Json::JsonNode root = doc.GetRoot();
   
   root.Set("type", "heartbeat");
   root.Set("timestamp", TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS));
   
   SendMessage(doc);
  }

//+------------------------------------------------------------------+
//| Send market data to Python                                      |
//+------------------------------------------------------------------+
void SendMarketData()
  {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
     {
      return;
     }

// Create JSON document
   MQL5_Json::JsonDocument doc = MQL5_Json::JsonNewObject();
   MQL5_Json::JsonNode root = doc.GetRoot();
   
   root.Set("type", "market_data");
   root.Set("timestamp", TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS));
   root.Set("symbol", _Symbol);
   root.Set("bid", tick.bid);
   root.Set("ask", tick.ask);
   root.Set("volume", (double)tick.volume);
   root.Set("spread", (tick.ask - tick.bid) / _Point);

// Add recent bars as array
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, 20, rates);
   if(copied > 0)
     {
      MQL5_Json::JsonNode bars = root.SetArray("bars");
      
      for(int i = 0; i < copied; i++)
        {
         MQL5_Json::JsonNode bar = bars.Add();
         bar.Set("time", TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS));
         bar.Set("open", rates[i].open);
         bar.Set("high", rates[i].high);
         bar.Set("low", rates[i].low);
         bar.Set("close", rates[i].close);
         bar.Set("volume", (double)rates[i].tick_volume);
        }
     }

   SendMessage(doc);
  }

//+------------------------------------------------------------------+
//| Request pending signals from Python                             |
//+------------------------------------------------------------------+
void RequestPendingSignals()
  {
   MQL5_Json::JsonDocument doc = MQL5_Json::JsonNewObject();
   MQL5_Json::JsonNode root = doc.GetRoot();
   
   root.Set("type", "request_signals");
   root.Set("timestamp", TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS));
   
   SendMessage(doc);
  }

//+------------------------------------------------------------------+
//| Process incoming messages from Python                           |
//+------------------------------------------------------------------+
void ProcessIncomingMessages()
  {
   if(socket_handle == INVALID_HANDLE || !is_connected)
     {
      return;
     }

// Read available data
   uchar temp_buffer[4096];
   int received = PRTF(SocketRead(socket_handle, temp_buffer, 4096, 0));

   if(received == -1)
     {
      int error = GetLastError();
      if(error == 4014)
        {
         // Normal - no data available yet
         return;
        }
      else if(error != 0)
        {
         is_connected = false;
        }
      return;
     }

   if(received == 0)
     {
      return;
     }

// Append to buffer
   int old_size = buffer_size;
   buffer_size += received;
   ArrayResize(message_buffer, buffer_size);
   ArrayCopy(message_buffer, temp_buffer, old_size, 0, received);

// Process complete messages
   while(buffer_size >= 4)
     {
      // Read length prefix (little-endian)
      uint message_length =
         message_buffer[0] |
         (message_buffer[1] << 8) |
         (message_buffer[2] << 16) |
         (message_buffer[3] << 24);

      // Check if we have complete message
      if(buffer_size < (int)(4 + message_length))
        {
         break;  // Wait for more data
        }

      // Extract message
      uchar msg_data[];
      ArrayResize(msg_data, (int)message_length);
      ArrayCopy(msg_data, message_buffer, 0, 4, (int)message_length);

      string message = CharArrayToString(msg_data, 0, (int)message_length, CP_UTF8);

      // Remove processed message from buffer
      int remaining = buffer_size - (4 + (int)message_length);
      if(remaining > 0)
        {
         uchar temp[];
         ArrayResize(temp, remaining);
         ArrayCopy(temp, message_buffer, 0, 4 + (int)message_length, remaining);
         ArrayResize(message_buffer, remaining);
         ArrayCopy(message_buffer, temp, 0, 0, remaining);
         buffer_size = remaining;
        }
      else
        {
         ArrayResize(message_buffer, 0);
         buffer_size = 0;
        }

      // Process message
      ProcessMessage(message);
     }
  }

//+------------------------------------------------------------------+
//| Process single JSON message from Python                         |
//+------------------------------------------------------------------+
void ProcessMessage(string json_string)
  {
   MQL5_Json::JsonError error;
   MQL5_Json::JsonDocument doc = MQL5_Json::JsonParse(json_string, error);
   
   if(!doc.IsValid())
     {
      Print("Failed to parse JSON: ", error.message);
      return;
     }

   MQL5_Json::JsonNode root = doc.GetRoot();
   string msg_type = root["type"].AsString("");

   if(msg_type == "signal")
     {
      ProcessSignal(root);
     }
   else if(msg_type == "heartbeat_response")
     {
      // Connection alive - no action needed
     }
   else
     {
      Print("Unknown message type: ", msg_type);
     }
  }

//+------------------------------------------------------------------+
//| Process trading signal from Python                              |
//+------------------------------------------------------------------+
void ProcessSignal(MQL5_Json::JsonNode &root)
  {
// Check if data field exists
   if(!root["data"].IsObject())
     {
      Print("Signal message missing 'data' field or it's not an object");
      return;
     }

   MQL5_Json::JsonNode data = root["data"];

// Extract signal fields with defaults
   string signal_type = data["signal_type"].AsString("");
   string symbol = data["symbol"].AsString("");
   double entry_price = data["entry_price"].AsDouble(0.0);
   double stop_loss = data["stop_loss"].AsDouble(0.0);
   double take_profit = data["take_profit"].AsDouble(0.0);
   double position_size = data["position_size"].AsDouble(0.0);
   string strategy_name = data["strategy_name"].AsString("unknown");

   if(signal_type == "" || symbol == "")
     {
      Print("Missing required fields in signal data");
      return;
     }

   Print("Received ", signal_type, " signal for ", symbol,
         " from strategy: ", strategy_name);

   if(!EnableTrading)
     {
      Print("Trading disabled - signal ignored");
      SendExecutionReport(signal_type, "ignored", 0.0);
      return;
     }

// Execute signal
   bool result = false;
   double execution_price = 0.0;

   if(signal_type == "BUY")
     {
      result = ExecuteBuy(symbol, position_size, stop_loss, take_profit);
      execution_price = SymbolInfoDouble(symbol, SYMBOL_ASK);
     }
   else if(signal_type == "SELL")
     {
      result = ExecuteSell(symbol, position_size, stop_loss, take_profit);
      execution_price = SymbolInfoDouble(symbol, SYMBOL_BID);
     }
   else if(signal_type == "CLOSE")
     {
      result = ClosePosition(symbol);
      execution_price = 0.0;
     }

// Send execution report
   string status = result ? "executed" : "failed";
   SendExecutionReport(signal_type, status, execution_price);
  }

//+------------------------------------------------------------------+
//| Execute buy order                                               |
//+------------------------------------------------------------------+
bool ExecuteBuy(string symbol, double size, double sl, double tp)
  {
   double volume = CalculateVolume(size);
   double price = SymbolInfoDouble(symbol, SYMBOL_ASK);

// Normalize SL/TP
   if(sl > 0)
      sl = NormalizeDouble(sl, _Digits);
   if(tp > 0)
      tp = NormalizeDouble(tp, _Digits);

   bool result = trade.Buy(volume, symbol, price, sl, tp, "Python Signal");

   if(result)
     {
      Print("BUY order executed: ", volume, " lots at ", price);
     }
   else
     {
      Print("BUY order failed: ", trade.ResultRetcodeDescription());
     }

   return result;
  }

//+------------------------------------------------------------------+
//| Execute sell order                                              |
//+------------------------------------------------------------------+
bool ExecuteSell(string symbol, double size, double sl, double tp)
  {
   double volume = CalculateVolume(size);
   double price = SymbolInfoDouble(symbol, SYMBOL_BID);

// Normalize SL/TP
   if(sl > 0)
      sl = NormalizeDouble(sl, _Digits);
   if(tp > 0)
      tp = NormalizeDouble(tp, _Digits);

   bool result = trade.Sell(volume, symbol, price, sl, tp, "Python Signal");

   if(result)
     {
      Print("SELL order executed: ", volume, " lots at ", price);
     }
   else
     {
      Print("SELL order failed: ", trade.ResultRetcodeDescription());
     }

   return result;
  }

//+------------------------------------------------------------------+
//| Close position for symbol                                       |
//+------------------------------------------------------------------+
bool ClosePosition(string symbol)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
        {
         if(PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
           {
            bool result = trade.PositionClose(ticket);
            if(result)
              {
               Print("Position closed: ", ticket);
              }
            else
              {
               Print("Failed to close position: ", trade.ResultRetcodeDescription());
              }
            return result;
           }
        }
     }

   Print("No position found for ", symbol);
   return false;
  }

//+------------------------------------------------------------------+
//| Calculate position volume based on risk                         |
//+------------------------------------------------------------------+
double CalculateVolume(double position_size)
  {
   if(position_size > 0)
     {
      // Use provided size
      return NormalizeDouble(position_size, 2);
     }

// Calculate based on risk percentage
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = balance * RiskPercent / 100.0;

// Simple volume calculation (can be enhanced)
   double volume = 0.01;  // Minimum volume

// Normalize to broker's allowed values
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   volume = MathMax(volume, min_volume);
   volume = MathMin(volume, max_volume);
   volume = MathFloor(volume / volume_step) * volume_step;

   return NormalizeDouble(volume, 2);
  }

//+------------------------------------------------------------------+
//| Send execution report to Python                                 |
//+------------------------------------------------------------------+
void SendExecutionReport(string signal_type, string status, double execution_price)
  {
   MQL5_Json::JsonDocument doc = MQL5_Json::JsonNewObject();
   MQL5_Json::JsonNode root = doc.GetRoot();
   
   root.Set("type", "execution_report");
   root.Set("timestamp", TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS));
   root.Set("signal_id", IntegerToString(TimeCurrent()));
   root.Set("signal_type", signal_type);
   root.Set("status", status);
   root.Set("execution_price", execution_price);
   root.Set("symbol", _Symbol);
   root.Set("magic_number", (double)MagicNumber);
   
   SendMessage(doc);
  }
//+------------------------------------------------------------------+