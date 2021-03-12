#ifndef _SSPUSDK_H
#define _SSPUSDK_H
#include <stdint.h>
#ifdef MYLIBDLL
#define MYLIBDLL  _declspec(dllimport) 
#else
#define MYLIBDLL  _declspec(dllexport) 
#endif
#define  ENABLE    0x01
#define  DISABLE   0x00


class SspuImp;
class SspuSDK
{
public:
	MYLIBDLL SspuSDK(void);
	MYLIBDLL virtual ~SspuSDK(void);
	//data collection start and stop functions (called by outside to manage the device)
	MYLIBDLL int configure();
// 	int loadConfiguration();
	MYLIBDLL int start();
	MYLIBDLL int stop();
	MYLIBDLL int open();
	MYLIBDLL int CamPortCmd(uint8_t portnum,uint8_t flag);
	MYLIBDLL int setCamBasePeriod(uint16_t period);
	MYLIBDLL int setCamParameter(uint8_t portnum,uint8_t timeperiod,uint8_t trigerdelay_ms);
	MYLIBDLL int setCamTriggerEdge(uint8_t triggeredge);
	MYLIBDLL int getCamTimeStamp(uint8_t portnum,long long *timestamp);
	MYLIBDLL int getCam1TimeStamp(long long *timestamp);
	MYLIBDLL int getCam2TimeStamp(long long *timestamp);
	MYLIBDLL int getCam3TimeStamp(long long *timestamp);
	MYLIBDLL int getCam4TimeStamp(long long *timestamp);
	MYLIBDLL int getCam5TimeStamp(long long *timestamp);
	MYLIBDLL int getCam6TimeStamp(long long *timestamp);
private:
	SspuImp* pr;

};

#endif