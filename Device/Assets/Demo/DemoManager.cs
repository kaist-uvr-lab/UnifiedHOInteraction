using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

public class DemoManager : MonoBehaviour
{
    // List<string> Gestures = new List<string>(new string[] {
    // "cclock_index",   // 0
    // "cclock_thumb",   // 1
    // "clock_index",    // 2
    // "clock_thumb",    // 3
    // "down_index",     // 4
    // "down_thumb",     // 5
    // "left_index",     // 6
    // "left_thumb",     // 7
    // "natural",        // 8
    // "right_index",    // 9
    // "right_thumb",    // 10
    // "tap_index",      // 11
    // "tap_thumb",      // 12
    // "up_index",       // 13
    // "up_thumb"        // 14
    // });

    List<string> Gestures = new List<string>(new string[] {
    "cclock",   // 0
    "cclock",   // 1
    "clock",    // 2
    "clock",    // 3
    "down",     // 4
    "down",     // 5
    "left",     // 6
    "left",     // 7
    "natural",  // 8
    "right",    // 9
    "right",    // 10
    "tap",      // 11
    "tap",      // 12
    "up",       // 13
    "up"        // 14
    });



    public GameObject PannelApp;
    public GameObject MusicApp;
    private CanvasGroup musicCanvasGroup;
    private CanvasGroup panelCanvasGroup;


    Dictionary<string, Button> pannelDict = new Dictionary<string, Button>();
    
    Dictionary<string, GameObject> musicDict = new Dictionary<string, GameObject>();
    public List<Texture> TextureList = new List<Texture>();

    Button curButton;
    int curIndex = 5;
    string InputString = "";
	private RawImage img;
    int textureIdx = 0;
    
    int demo_step_idx = 0;

    string mode = "init";

    GameObject speaker;
    Slider slider;

    // parameters
    float baseBounceStrength = 0.1f;
    float bounceStrength = 0.1f; // 바운스 크기
    float baseBounceSpeed = 2.5f; 
    float bounceSpeed = 2.5f; 
    float fadeDuration = 0.5f;
    


    
    // Start is called before the first frame update
    void Start()
    {
        // PannelApp.SetActive(false);
        // MusicApp.SetActive(false);

        panelCanvasGroup = PannelApp.GetComponent<CanvasGroup>();
        if (panelCanvasGroup == null)
        {
            Debug.LogError("PannelApp에 CanvasGroup이 없습니다. Inspector에서 추가해주세요!");
        }
        musicCanvasGroup = MusicApp.GetComponent<CanvasGroup>();
        if (musicCanvasGroup == null)
        {
            Debug.LogError("MusicApp에 CanvasGroup이 없습니다. Inspector에서 추가해주세요!");
        }



        // 원하는 색상 정의
        Color normalColor     = Color.white;                        
        Color pressedColor    = new Color(0.90f, 0.20f, 0.20f, 1f); // 선명한 빨강 (#E53935 근사)
        Color selectedColor= new Color(1.00f, 0.92f, 0.23f, 1f); // 밝은 노랑 (#FFEB3B 근사)



        //get child by name under Pannel
        Transform pannel = PannelApp.transform.Find("Pannel");

        foreach (Transform child in pannel)
        {
            pannelDict[child.name] = child.gameObject.GetComponent<UnityEngine.UI.Button>();

            var colors = pannelDict[child.name].colors;
            colors.normalColor     = normalColor;
            colors.selectedColor   = selectedColor;
            colors.pressedColor    = pressedColor;
            pannelDict[child.name].colors = colors;

        }
        curButton = pannelDict[curIndex.ToString()];
        curButton.Select();


        //get child by name under Music 

        Transform Canvas = MusicApp.transform.Find("Canvas");
        foreach (Transform child in Canvas)
        {
            musicDict[child.name] = child.gameObject;
        }
        img = musicDict["Album"].GetComponent<RawImage>();        
        img.texture = (Texture)TextureList[textureIdx];

        
        speaker = MusicApp.transform.Find("Speaker").gameObject;
        slider = musicDict["Slider"].GetComponent<Slider>();
        
    }

    // Update is called once per frame
    void Update()
    {
        if (MusicApp.activeSelf)
        {
            BounceSpeaker();
        }
        
        if (mode == "init")
            switch (InputString)
            {         
                case "tap":
                    if (demo_step_idx == 0)
                    {
                        demo_step_idx = 1;
                        ShowApp(PannelApp);

                        pannelDict[curIndex.ToString()].Select();                        

                        mode = "pannel";
                    }
                    else
                    {
                        ShowApp(MusicApp);
                        mode = "music";                        
                    }
                    
                    InputString = "";
                    break;
                default:
                    break;
            }
        
        if (mode == "pannel")
            switch (InputString)
            {
                case "up":
                    Move(Vector2.up);
                    InputString = "";                
                    break;
                case "down":
                    Move(Vector2.down);
                    InputString = "";
                    break;
                case "left":
                    Move(Vector2.left);
                    InputString = "";
                    break;
                case "right":
                    Move(Vector2.right);
                    InputString = "";
                    break;
                case "tap":                
                    pannelDict[curIndex.ToString()].onClick.Invoke();

                    var colors = pannelDict[curIndex.ToString()].colors;
                    pannelDict[curIndex.ToString()].targetGraphic.color = colors.pressedColor;
                    StartCoroutine(ResetColor(pannelDict[curIndex.ToString()], colors.disabledColor, 0.4f));

                    mode = "music";
                    StartCoroutine(HidePanelAfterDelay(PannelApp, 0.4f));

                    InputString = "";
                    break;
            }
        

        if (mode == "music")
            switch (InputString)
            {
                case "tap":
                    HideApp(MusicApp);

                    mode = "init";
                    InputString = "";
                    break;                
                case "cclock":
                    StartCoroutine(DecreaseVolume());
                    InputString = "";
                    break;
                case "clock":
                    StartCoroutine(IncreaseVolume());
                    InputString = "";
                    break;
                case "down":
                    StartCoroutine(DecreaseVolume());
                    InputString = "";
                    break;
                case "up":
                    StartCoroutine(IncreaseVolume());
                    InputString = "";
                    break;
                case "left":
                    textureIdx -= 1;
                    if (textureIdx < 0)
                        textureIdx = 2;                                           
                    img = (RawImage)musicDict["Album"].GetComponent<RawImage>();
                    img.texture = (Texture)TextureList[textureIdx];
                    
                    InputString = "";
                    break;
                case "right":
                    textureIdx += 1;
                    if (textureIdx > 2)
                        textureIdx = 0;
                             
                    img = (RawImage)musicDict["Album"].GetComponent<RawImage>();
                    img.texture = (Texture)TextureList[textureIdx];
                    
                    InputString = "";
                    break;
                
            }


        // // For debugging with keyboard input
        // // Keypad 4 → Left
        // if (Input.GetKeyUp(KeyCode.Keypad4) || Input.GetKeyUp(KeyCode.Alpha4))
        // {
        //     InputString = "left";
        //     Debug.Log("입력: " + InputString);
        // }

        // // Keypad 6 → Right
        // if (Input.GetKeyUp(KeyCode.Keypad6) || Input.GetKeyUp(KeyCode.Alpha6))
        // {
        //     InputString = "right";
        //     Debug.Log("입력: " + InputString);
        // }

        // // Keypad 8 → Up
        // if (Input.GetKeyUp(KeyCode.Keypad8) || Input.GetKeyUp(KeyCode.Alpha8))
        // {
        //     InputString = "up";
        //     Debug.Log("입력: " + InputString);
        // }

        // // Keypad 2 → Down
        // if (Input.GetKeyUp(KeyCode.Keypad2) || Input.GetKeyUp(KeyCode.Alpha2))
        // {
        //     InputString = "down";
        //     Debug.Log("입력: " + InputString);
        // }
        // // Keypad 5 → Tap
        // if (Input.GetKeyUp(KeyCode.Keypad5) || Input.GetKeyUp(KeyCode.Alpha5))
        // {
        //     InputString = "tap";
        //     Debug.Log("입력: " + InputString);
        // }
        // // Keypad 9 → clock
        // if (Input.GetKeyUp(KeyCode.Keypad9) || Input.GetKeyUp(KeyCode.Alpha9))
        // {
        //     InputString = "clock";
        //     Debug.Log("입력: " + InputString);
        // }
        // // Keypad 7 → cclock
        // if (Input.GetKeyUp(KeyCode.Keypad7) || Input.GetKeyUp(KeyCode.Alpha7))
        // {
        //     InputString = "cclock";
        //     Debug.Log("입력: " + InputString);
        // }


    }

    void Move(Vector2 dir)
    {
        // curIndex는 1~9 범위
        int row = 2 - ((curIndex - 1) / 3); // 키패드 배열 기준
        int col = (curIndex - 1) % 3;

        int newRow = row;
        int newCol = col;

        if (dir == Vector2.up)    newRow--;
        if (dir == Vector2.down)  newRow++;
        if (dir == Vector2.left)  newCol--;
        if (dir == Vector2.right) newCol++;

        // wrap-around 처리
        if (newRow < 0) newRow = 2;
        if (newRow > 2) newRow = 0;
        if (newCol < 0) newCol = 2;
        if (newCol > 2) newCol = 0;

        // 다시 1~9로 변환
        curIndex = (2 - newRow) * 3 + newCol + 1;

        Debug.Log("curIndex: " + curIndex);
        pannelDict[curIndex.ToString()].Select();
    }

    private IEnumerator ResetColor(Button btn, Color normalColor, float delay)
    {
        yield return new WaitForSeconds(delay);
        btn.targetGraphic.color = normalColor;
    }
    IEnumerator HidePanelAfterDelay(GameObject App, float delay)
    {
        yield return new WaitForSeconds(delay);
        HideApp(App);

    }

    
    public void event_ActivateApp(GameObject TargetApp)
    {
        Debug.Log("App Activated");
        ShowApp(TargetApp);


    }

    void BounceSpeaker()
    {
        
        float noise = (Mathf.PerlinNoise(Time.time, 0f) - 0.5f) * 0.1f; 

        float t = Mathf.PingPong(Time.time * (bounceSpeed + noise), 1f);
        float spike = Mathf.Exp(-5f * t) * 0.5f;

        float scale = 1f + spike * bounceStrength;
        speaker.transform.localScale = new Vector3(scale, scale, scale);

    }
    
    IEnumerator IncreaseVolume()
    {
        for (int i =0; i < 40; i++)
        {
            if (slider.value < 1)
            {
                slider.value += 0.005f;
                yield return new WaitForSeconds(0.01f);
            }
        }    
        
        bounceStrength = baseBounceStrength + slider.value * 0.2f;
        bounceSpeed = baseBounceSpeed + slider.value * 3.0f;
        
    }

    IEnumerator DecreaseVolume()
    {
        for (int i = 0; i < 40; i++)
        {
            if (slider.value > 0)
            {
                slider.value -= 0.005f;
                yield return new WaitForSeconds(0.01f);
            }
        }
        bounceStrength = baseBounceStrength + slider.value * 0.2f;
        bounceSpeed = baseBounceSpeed + slider.value * 3.0f;
    }

    // 특정 앱을 페이드 인
    public void ShowApp(GameObject app)
    {
        app.SetActive(true);
        CanvasGroup cg = app.GetComponent<CanvasGroup>();
        if (cg != null)
        {
            StartCoroutine(FadeCanvasGroup(cg, 0f, 1f));
        }
    }

    // 특정 앱을 페이드 아웃
    public void HideApp(GameObject app)
    {
        CanvasGroup cg = app.GetComponent<CanvasGroup>();
        if (cg != null)
        {
            StartCoroutine(FadeOutAndDisable(app, cg));
        }
    }

    private IEnumerator FadeCanvasGroup(CanvasGroup cg, float start, float end)
    {
        float time = 0f;
        while (time < fadeDuration)
        {
            cg.alpha = Mathf.Lerp(start, end, time / fadeDuration);
            time += Time.deltaTime;
            yield return null;
        }
        cg.alpha = end;
    }

    private IEnumerator FadeOutAndDisable(GameObject app, CanvasGroup cg)
    {
        yield return FadeCanvasGroup(cg, 1f, 0f);
        app.SetActive(false);
    }


    public void GetInputMessage(double[] msgPose, int msgGesture)
    {
        float[] handPose3D = msgPose.Select(x => (float)x).ToArray();

        InputString = Gestures[msgGesture];

    }
}
