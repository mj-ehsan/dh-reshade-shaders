#include "Reshade.fxh"

#ifndef DH_RESOLUTION_SCALE
 #define DH_RESOLUTION_SCALE 0.5
#endif

// Can be used to fix wrong screen resolution
#define INPUT_WIDTH BUFFER_WIDTH
#define INPUT_HEIGHT BUFFER_HEIGHT

#define RENDER_WIDTH INPUT_WIDTH*DH_RESOLUTION_SCALE
#define RENDER_HEIGHT INPUT_HEIGHT*DH_RESOLUTION_SCALE

#define RENDER_SIZE int2(RENDER_WIDTH,RENDER_HEIGHT)

#define BUFFER_SIZE int2(INPUT_WIDTH,INPUT_HEIGHT)
#define BUFFER_SIZE3 int3(INPUT_WIDTH,INPUT_HEIGHT,INPUT_WIDTH*RESHADE_DEPTH_LINEARIZATION_FAR_PLANE*fDepthMultiplier/1024)
#define NOISE_SIZE 512

#define PI 3.14159265359

#define getDepth(c) ReShade::GetLinearizedDepth(c)
#define getNormal(c) (tex2Dlod(normalSampler,float4(c.xy,0,0)).xyz-0.5)*2
	#define getColorSampler(s,c) tex2Dlod(s,float4(c.xy,0,0))

#define getColor(c) tex2Dlod(ReShade::BackBuffer,float4(c.xy,0,0))

#define diffT(v1,v2,t) !any(max(abs(v1-v2)-t,0))

namespace DHRTGI {

    texture blueNoiseTex < source ="dh_rt_noise.png" ; > { Width = NOISE_SIZE; Height = NOISE_SIZE; MipLevels = 1; Format = RGBA8; };
    sampler blueNoiseSampler { Texture = blueNoiseTex;  AddressU = REPEAT;  AddressV = REPEAT;  AddressW = REPEAT;};

    texture normalTex { Width = INPUT_WIDTH; Height = INPUT_HEIGHT; Format = RGBA8; };
    sampler normalSampler { Texture = normalTex;};
    
    texture lightPassTex { Width = RENDER_WIDTH; Height = RENDER_HEIGHT; Format = RGBA8; MipLevels = 4; };
    sampler lightPassSampler { Texture = lightPassTex; MinLOD = 0.0f; MaxLOD = 3.0f;};
    
    texture lightPassHitTex { Width = RENDER_WIDTH; Height = RENDER_HEIGHT; Format = RGBA8; };
    sampler lightPassHitSampler { Texture = lightPassHitTex; };

    texture smoothPass3Tex { Width = RENDER_WIDTH; Height = RENDER_HEIGHT; Format = RGBA8; MipLevels = 4; };
    sampler smoothPass3Sampler { Texture = smoothPass3Tex; MinLOD = 0.0f; MaxLOD = 3.0f; };

    texture smoothAOPass3Tex { Width = RENDER_WIDTH; Height = RENDER_HEIGHT; Format = RGBA8; };
    sampler smoothAOPass3Sampler { Texture = smoothAOPass3Tex; MinLOD = 0.0f; MaxLOD = 3.0f; };
    
    texture resultTex { Width = INPUT_WIDTH; Height = INPUT_HEIGHT; Format = RGBA8; };
    sampler resultSampler { Texture = resultTex; };

    uniform float frametime < source = "frametime"; >;
    uniform int framecount < source = "framecount"; >;
    uniform int random < source = "random"; min = 0; max = NOISE_SIZE; >;

    uniform bool bDebug <
        ui_category = "Setting";
        ui_label = "Display light only";
    > = false;
    
    uniform float fDepthMultiplier <
        ui_type = "slider";
        ui_category = "Setting";
        ui_label = "Depth multiplier";
        ui_min = 0.1; ui_max = 10;
        ui_step = 0.1;
    > = 1.0;
    
    uniform float fSkyDepth <
        ui_type = "slider";
        ui_category = "Setting";
        ui_label = "Sky Depth ";
        ui_min = 0.0; ui_max = 1.0;
        ui_step = 0.01;
    > = 0.99;
    
    uniform float fWeaponDepth <
        ui_type = "slider";
        ui_category = "Setting";
        ui_label = "Weapon Depth ";
        ui_min = 0.0; ui_max = 1.0;
        ui_step = 0.001;
    > = 0.001;

// RAY TRACING
    #define iRayPreciseHit 2
    
	#define iRayPreciseHitSteps 2

    #define fRayStepPrecision 10000
    
    uniform float fRayStepMultiply <
        ui_type = "slider";
        ui_category = "Ray tracing";
        ui_label = "Step multiply";
        ui_min = 0.01; ui_max = 4.0;
        ui_step = 0.01;
    > = 1.0;

    uniform float fRayHitDepthThreshold <
        ui_type = "slider";
        ui_category = "Ray tracing";
        ui_label = "Ray Hit Depth Threshold";
        ui_min = 0.001; ui_max = 1;
        ui_step = 0.001;
    > = 0.500;
    
// LIGHT COLOR
    
    uniform float fRayBounce <
        ui_type = "slider";
        ui_category = "COLOR";
        ui_label = "Bounce strength";
        ui_min = 0; ui_max = 1.0;
        ui_step = 0.01;
    > = 0.25;
        
    uniform float fFadePower <
        ui_type = "slider";
        ui_category = "COLOR";
        ui_label = "Distance Fading";
        ui_min = 0.1; ui_max = 10;
        ui_step = 0.01;
    > = 2.5;
    
    uniform float fSaturateColor <
        ui_type = "slider";
        ui_category = "COLOR";
        ui_label = "Saturate";
        ui_min = 0.0; ui_max = 1.0;
        ui_step = 0.01;
    > = 0.10;
    
// AO
    uniform float fAOMultiplier <
        ui_type = "slider";
        ui_category = "AO";
        ui_label = "Multiplier";
        ui_min = 0.0; ui_max = 5;
        ui_step = 0.01;
    > = 1;
    
    uniform int iAODistance <
        ui_type = "slider";
        ui_category = "AO";
        ui_label = "Distance";
        ui_min = 0; ui_max = 1280;
        ui_step = 1;
    > = 6;
 
    
// MERGING
    
    uniform float fSourceColor <
        ui_type = "slider";
        ui_category = "Merging";
        ui_label = "Source color";
        ui_min = 0.1; ui_max = 2;
        ui_step = 0.01;
    > = 0.90;
    
    uniform float fLightMult <
        ui_type = "slider";
        ui_category = "Merging";
        ui_label = "Light multiplier";
        ui_min = 0.1; ui_max = 10;
        ui_step = 0.01;
    > = 1.0;
    
    uniform float fMaxLight <
        ui_type = "slider";
        ui_category = "Merging";
        ui_label = "MAx light";
        ui_min = 0.0; ui_max = 1.0;
        ui_step = 0.1;
    > = 1.0;
    
    uniform float fLightNormalize <
        ui_type = "slider";
        ui_category = "Merging";
        ui_label = "Light normalize";
        ui_min = 0.1; ui_max = 4;
        ui_step = 0.01;
    > = 0.1;
    

    
//////// COLOR SPACE
    float RGBCVtoHUE(in float3 RGB, in float C, in float V) {
        float3 Delta = (V - RGB) / C;
        Delta.rgb -= Delta.brg;
        Delta.rgb += float3(2,4,6);
        Delta.brg = step(V, RGB) * Delta.brg;
        float H;
        H = max(Delta.r, max(Delta.g, Delta.b));
        return frac(H / 6);
    }

    float3 RGBtoHSL(in float3 RGB) {
        float3 HSL = 0;
        float U, V;
        U = -min(RGB.r, min(RGB.g, RGB.b));
        V = max(RGB.r, max(RGB.g, RGB.b));
        HSL.z = ((V - U) * 0.5);
        float C = V + U;
        if (C != 0)
        {
            HSL.x = RGBCVtoHUE(RGB, C, V);
            HSL.y = C / (1 - abs(2 * HSL.z - 1));
        }
        return HSL;
    }
      
    float3 HUEtoRGB(in float H) 
    {
        float R = abs(H * 6 - 3) - 1;
        float G = 2 - abs(H * 6 - 2);
        float B = 2 - abs(H * 6 - 4);
        return saturate(float3(R,G,B));
    }
      
    float3 HSLtoRGB(in float3 HSL)
    {
        float3 RGB = HUEtoRGB(HSL.x);
        float C = (1 - abs(2 * HSL.z - 1)) * HSL.y;
        return (RGB - 0.5) * C + HSL.z;
    }

    float getBrightness(float3 color) {
        return max(max(color.r,color.g),color.b);
    }
////// COORDINATES
    
    float2 InputPixelSize() {
        float2 result = 1.0;
        return result/float2(INPUT_WIDTH,INPUT_HEIGHT);
    }
    
    float2 RenderPixelSize() {
        float2 result = 1.0;
        return result/float2(RENDER_WIDTH,RENDER_HEIGHT);
    }

    bool inScreen(float2 coords) {
        return coords.x>=0 && coords.x<=1
            && coords.y>=0 && coords.y<=1;
    }
    
    bool inScreen(float3 coords) {
        return coords.x>=0 && coords.x<=1
            && coords.y>=0 && coords.y<=1
            && coords.z>=0 && coords.z<=1;
    }

    float3 getWorldPosition(float2 coords) {
        float depth = getDepth(coords);
        float3 result = float3((coords-0.5)*depth,depth);
        result *= BUFFER_SIZE3;
        return result;
    }

    float3 getScreenPosition(float3 wp) {
        float3 result = wp/BUFFER_SIZE3;
        result.xy /= result.z;
        return float3(result.xy+0.5,result.z);
    }
    
    float noise1(float2 co)
	{
	  return frac(sin(dot(co.xy ,float2(1.0,73))) * 43758.5453);
	}
	
    float noise(float2 co)
	{
	  return noise1(noise1(noise1(co)));
	}
	uniform float Timer < source = "timer"; >;
	float3 noise3dts(float2 co, float s, bool t)
	{
		co += sin(Timer/5120.125245662)*t;
		co += s/16.3542625435332254;
		return float3( noise(co), noise(co+0.6432168421), noise(co+0.19216811));
	}

    float3 getNormalJitter(float2 coords) {
    
       float3 jitter = noise3dts( tex2D(blueNoiseSampler, coords).rg, 1, 1)-0.5;       
		return normalize(jitter);
        
    }
    
    float3 getColorBounce(float2 coords) {
        float3 result = getColor(coords).rgb;
        if(fRayBounce>0) {
            result = saturate(result+getColorSampler(resultSampler,coords).rgb*fRayBounce);
        }
        return result;
    }

    void PS_NormalPass(float4 vpos : SV_Position, float2 coords : TexCoord, out float4 outNormal : SV_Target0) {
        
        float3 offset = float3(ReShade::PixelSize, 0.0);

        float3 posCenter = getWorldPosition(coords);
        float3 posNorth  = getWorldPosition(coords - offset.zy);
        float3 posEast   = getWorldPosition(coords + offset.xz);
        float3 normal = normalize(cross(posCenter - posNorth, posCenter - posEast));
        
        float4 r = float4(normal/2.0+0.5,1.0);
        outNormal = r;
        
    }
    
    float3 getRayColor(float2 coords) {
        float3 color = getColorBounce(coords);
            
        if(fSaturateColor>0) {
            float3 hsl = RGBtoHSL(color);
            if(hsl.y>0.1 && hsl.z>0.1) {
                float maxChannel = getBrightness(color.rgb);
                if(maxChannel>0) {
                    float3 saturatedColor = color.rgb/maxChannel;
                    color.rgb = fSaturateColor*saturatedColor+(1.0-fSaturateColor)*color.rgb;
                }
            }
        }
        return color;
    }

    float4 trace(float3 refWp,float3 lightVector,float startDepth) {

        float stepRatio = 1.0+fRayStepMultiply/10.0;
        
        float stepLength = 1.0/fRayStepPrecision;
        float3 incrementVector = lightVector*stepLength;
        float traceDistance = 0;
        float3 currentWp = refWp;
        
        float rayHitIncrement = fRayHitDepthThreshold/50.0;
        float rayHitDepthThreshold = rayHitIncrement;

        bool crossed = false;
        float deltaZ = 0;
        float deltaZbefore = 0;
        float3 lastCross;
        
        bool outSource = false;
        bool firstStep = true;
        
        bool startWeapon = startDepth<fWeaponDepth;
        float weaponLimit = fWeaponDepth*BUFFER_SIZE3.z;
        
        
        do {
            currentWp += incrementVector;
            traceDistance += stepLength;
            
            float3 screenCoords = getScreenPosition(currentWp);
            
            bool outScreen = !inScreen(screenCoords) && (!startWeapon || currentWp.z<weaponLimit);
            
            float3 screenWp = getWorldPosition(screenCoords.xy);
            
            deltaZ = screenWp.z-currentWp.z;

            if(firstStep && deltaZ<=0) {

                // wrong direction
                float3 n = getNormal(getScreenPosition(refWp));
                incrementVector = reflect(incrementVector,n);
                
                currentWp = refWp+incrementVector;

                firstStep = false;
            } else if(outSource) {
                if(!outScreen && sign(deltaZ)!=sign(deltaZbefore)) {
                    // search more precise
                    float preciseRatio = 1.0/iRayPreciseHitSteps;
                    float3 preciseIncrementVector = incrementVector;
                    float preciseLength = stepLength;
                    for(int precisePass=0;precisePass<iRayPreciseHit;precisePass++) {
                        preciseIncrementVector *= preciseRatio;
                        preciseLength *= preciseRatio;
                        
                        int preciseStep=0;
                        bool recrossed=false;
                        if(!recrossed && preciseStep<iRayPreciseHitSteps-1) {
                            currentWp -= preciseIncrementVector;
                            traceDistance -= preciseLength;
                            deltaZ = screenWp.z-currentWp.z;
                            recrossed = sign(deltaZ)==sign(deltaZbefore);
                            preciseStep++;
                        }
                        if(recrossed) {
                            currentWp += preciseIncrementVector;
                            traceDistance += preciseLength;
                            deltaZ = screenWp.z-currentWp.z;
                        }
                    }
                    
                    lastCross = currentWp;
                    crossed = true;
                    
                    
                }
                if(abs(deltaZ)<=rayHitDepthThreshold || outScreen) {
                    // hit !
                    return float4(crossed ? lastCross : currentWp,1.0);
                }
            } else {
                if(outScreen) {
                    currentWp -= incrementVector;
                    return float4(currentWp,0.0);
                }
                outSource = abs(deltaZ)>rayHitDepthThreshold;
            }

            firstStep = false;
            
            deltaZbefore = deltaZ;
            
            stepLength *= stepRatio;
            if(rayHitDepthThreshold<fRayHitDepthThreshold) rayHitDepthThreshold +=rayHitIncrement;
            incrementVector *= stepRatio;

        } while(traceDistance<INPUT_WIDTH);

        return 0.0;

    }

    void PS_LightPass(float4 vpos : SV_Position, float2 coords : TexCoord, out float4 outColor : SV_Target0, out float4 outHit : SV_Target1, out float4 outDistance : SV_Target2) {
        
        float depth = getDepth(coords);
        if(depth>fSkyDepth) {
            outColor = 0; outDistance = 0; outHit = 0;
        } else {
        
        float3 targetWp = getWorldPosition(coords);
        float3 targetNormal = getNormalJitter(coords);

        float3 lightVector = reflect(targetWp,targetNormal);
        
        float opacity = 1.0;
        
        float4 hitPosition = trace(targetWp,lightVector,depth);
        
        
        float3 screenCoords = getScreenPosition(hitPosition.xyz);
        float3 color = getRayColor(screenCoords.xy);
        if(hitPosition.a==0) {
            // no hit
            outColor = float4(color,opacity);
          outHit = float4(screenCoords,1);  
           outDistance = float4(1,0,0,opacity/4);
        } else {
            float b = getBrightness(color);
            
            float d = abs(distance(hitPosition.xyz,targetWp));
                        
            float distance = 1.0+0.02*d;
            float distanceRatio = 0.1+1.0/pow(distance,0.05*fFadePower);
            
            outColor = float4(hitPosition.a*distanceRatio*color,opacity);
          outHit = float4(screenCoords,1);  
            outDistance = depth<fWeaponDepth ? float4(1,0,0,1) : float4(d>iAODistance ? 1 : b+(d/iAODistance),0,0,opacity);
        }}
    }
    
    float3 max3(float3 a,float3 b) {
        return float3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));
    }
    
    float3 min3(float3 a,float3 b) {
        return float3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));
    }
    
    float3 minSum3(float3 a,float b) {
        float s = a.x+a.y+a.z;
        return s>0 && s>b ? b*a/s : a;
    }
    
    void PS_UpdateResult(in float4 position : SV_Position, in float2 coords : TEXCOORD, out float4 outPixel : SV_Target,out float4 outDepth : SV_Target1)
    {
        float3 color = getColor(coords).rgb;
        float depth = getDepth(coords);
        
        float3 colorHsl = RGBtoHSL(color);
        float3 light = getColorSampler(smoothPass3Sampler,coords).rgb;
        float3 lightHsl = RGBtoHSL(light);
        
        float b = getBrightness(color);
        float lb = getBrightness(light);
        
        float ao = getColorSampler(smoothAOPass3Sampler,coords).r;
        ao = pow(abs(ao), fAOMultiplier);

        float3 colorDesatHsl = colorHsl;

        colorDesatHsl = saturate(colorDesatHsl);
        float3 colorDesat = HSLtoRGB(colorDesatHsl);
        float3 lightApply = light*colorDesat;
        
        float3 colorHueShift = colorHsl;
        colorHueShift.x = lightHsl.x;
        colorHueShift.y = lightHsl.y;
        colorHueShift.z = pow(colorHueShift.z+0.1,0.5)*abs(sin(lightHsl.z*PI));
        lightApply = HSLtoRGB(colorHueShift);

       float colorRatio = fSourceColor;
       float lightRatio = fLightMult*(1.0-b)+lightHsl.y+lightHsl.z;


       float mLight = min(b*2+0.1,fMaxLight);
       float3 result = (1.0-b)*2*(colorHueShift.y)*lightApply+(1.0-colorHueShift.y)*color;
       result = (color*colorRatio+minSum3(fLightMult*light,mLight)*result*lightRatio)/(0.9+fLightNormalize);
       result *= ao;

       outPixel = float4(saturate(result),1.0);
       
       
       outDepth = float4(getDepth(coords),0,0,1);
    }
    
    void PS_DisplayResult(in float4 position : SV_Position, in float2 coords : TEXCOORD, out float4 outPixel : SV_Target)
    {
        float4 color = getColorSampler(resultSampler,coords);
        if(bDebug) {
            float b = getBrightness(color.rgb);
            float ao = max(getColorSampler(smoothAOPass3Sampler,coords).r,b);
            color = ao * getColorSampler(smoothPass3Sampler,coords);
            color.a = 1;
        }
        outPixel = color;
    }
    
    
    technique DH_RTGI {
        pass {
            VertexShader = PostProcessVS;
            PixelShader = PS_NormalPass;
            RenderTarget = normalTex;
        }
        pass {
            VertexShader = PostProcessVS;
            PixelShader = PS_LightPass;
            RenderTarget = smoothPass3Tex;
            RenderTarget1 = lightPassHitTex;
            RenderTarget2 = smoothAOPass3Tex;
        }
        pass {
            VertexShader = PostProcessVS;
            PixelShader = PS_UpdateResult;
            RenderTarget = resultTex;
        }
        pass {
            VertexShader = PostProcessVS;
            PixelShader = PS_DisplayResult;
        }
    }


}
