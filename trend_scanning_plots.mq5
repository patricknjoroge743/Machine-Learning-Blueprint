#property strict
#include <Canvas\Canvas.mqh>

CCanvas canvas;

//--- parameters
input int MinHorizon = 5;
input int MaxHorizon = 20;
input int BarsBack   = 200;
input color UpColor   = clrGreen;
input color DownColor = clrRed;
input color FlatColor = clrGray;

//--- function: linear trend t-stat
double TStatLinearTrend(const double &y[])
{
   int n = ArraySize(y);
   if(n < 3) return 0.0;

   double sumX=0, sumY=0, sumXY=0, sumXX=0;
   for(int i=0;i<n;i++)
   {
      sumX  += i;
      sumY  += y[i];
      sumXY += i*y[i];
      sumXX += i*i;
   }
   double meanX=sumX/n;
   double meanY=sumY/n;

   double slope_num=0, slope_den=0;
   for(int i=0;i<n;i++)
   {
      slope_num += (i-meanX)*(y[i]-meanY);
      slope_den += (i-meanX)*(i-meanX);
   }
   double slope = slope_num/slope_den;

   // error term and t-value
   double SSE=0;
   for(int i=0;i<n;i++)
   {
      double yhat = meanY + slope*(i-meanX);
      SSE += (y[i]-yhat)*(y[i]-yhat);
   }
   double sigma2 = SSE/(n-2);
   double se_slope = MathSqrt(sigma2/slope_den);

   if(se_slope==0) return 0.0;
   return slope/se_slope;
}

//--- structure for labels
struct TrendLabel
{
   int startIdx;
   int endIdx;
   int label;   // -1, 0, +1
};

TrendLabel labels[];

//--- create canvas
int OnInit()
{
   if(!canvas.CreateBitmapLabel("Canvas",0,0,800,400,COLOR_FORMAT_ARGB_NORMALIZE))
   {
      Print("Canvas creation failed");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}

void OnStart()
{
   // 1. Get close prices
   double close[];
   ArraySetAsSeries(close,true);
   CopyClose(_Symbol,PERIOD_M5,0,BarsBack,close);

   // 2. Run trend-scanning
   ArrayFree(labels);
   for(int i=MaxHorizon; i<BarsBack; i++)
   {
      double bestT=0; int bestH=0;
      for(int h=MinHorizon; h<=MaxHorizon; h++)
      {
         if(i-h<0) continue;
         double seg[];
         ArrayResize(seg,h);
         for(int j=0;j<h;j++) seg[j]=close[i-j];
         double t=TStatLinearTrend(seg);
         if(MathAbs(t)>MathAbs(bestT)) { bestT=t; bestH=h; }
      }
      if(bestH>0)
      {
         TrendLabel lbl;
         lbl.startIdx=i;
         lbl.endIdx  =i-bestH+1;
         lbl.label=(bestT>0)?1:((bestT<0)?-1:0);
         ArrayPush(labels,lbl);
      }
   }

   // 3. Draw chart
   canvas.Erase(clrWhite);

   int W=canvas.Width();
   int H=canvas.Height();

   int left=50,right=20,top=20,bottom=30;
   int plotW=W-left-right;
   int plotH=H-top-bottom;

   double minP=close[ArrayMinimum(close,0,BarsBack)];
   double maxP=close[ArrayMaximum(close,0,BarsBack)];
   double scaleY=plotH/(maxP-minP+1e-6);

   double stepX=(double)plotW/(BarsBack-1);

   // draw price line
   for(int i=0;i<BarsBack-1;i++)
   {
      int x1=left+(int)(i*stepX);
      int x2=left+(int)((i+1)*stepX);
      int y1=H-bottom-(int)((close[i]-minP)*scaleY);
      int y2=H-bottom-(int)((close[i+1]-minP)*scaleY);
      canvas.Line(x1,y1,x2,y2,clrBlack);
   }

   // overlay trend highlights
   for(int k=0;k<ArraySize(labels);k++)
   {
      TrendLabel lbl=labels[k];
      int x1=left+(int)((BarsBack-1-lbl.startIdx)*stepX);
      int x2=left+(int)((BarsBack-1-lbl.endIdx)*stepX);
      int y0=H-bottom;
      int y1=top;

      color c=FlatColor;
      if(lbl.label==1) c=UpColor;
      if(lbl.label==-1) c=DownColor;

      canvas.FillRectangle(x1,y1,x2,y0,c,80);
   }

   // axes
   canvas.Line(left,H-bottom,W-right,H-bottom,clrBlack);
   canvas.Line(left,top,left,H-bottom,clrBlack);

   canvas.Update();
}