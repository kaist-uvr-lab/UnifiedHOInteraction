using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;

public class DemoManager_backup : MonoBehaviour
{
    public GameObject DemoApp;
    public GameObject PannelApp;
    public GameObject MusicApp;

    Dictionary<string, Button> pannelDict = new Dictionary<string, Button>();
    
    Dictionary<string, GameObject> musicDict = new Dictionary<string, GameObject>();
    public List<Texture> TextureList = new List<Texture>();

    Button curButton;
    int curIndex = 5;
    string InputString = "";
	private RawImage img;
    int textureIdx = 0;
    
    int demo_step_idx = 0;





    

    private float _currentScale = 1000.0f;
    private float InitScale = 1000.0f;
    private const int FramesCount = 100;
    private const float AnimationTimeSeconds = 1.0f;
    private float varAnimTime = 0.8f;
 
    private bool _upScale = true;

    private bool scaling = false;
    private float maxScale = 50.0f;

    bool flag_activate = false;

    // Start is called before the first frame update
    void Start()
    {
        // PannelApp.SetActive(false);
        // MusicApp.SetActive(false);

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
        GameObject speaker = MusicApp.transform.Find("Speaker").gameObject;

        Transform Canvas = PannelApp.transform.Find("Canvas");
        foreach (Transform child in Canvas.transform)
        {
            musicDict[child.name] = child.gameObject;
        }
        img = musicDict["Album"].GetComponent<RawImage>();
        
        img.texture = (Texture)TextureList[textureIdx];
        
    }

    // Update is called once per frame
    void Update()
    {
        // if (mode == "select")
        // {
        //     switch (InputString)
        //     {
        //         case "up":
        //             NormalButton(curButton);
        //             InputString = "";
        //             curIndex -= 3;
        //             if (curIndex <= 0)
        //                 curIndex += 9;
        //             curButton = gridDict[curIndex.ToString()];
        //             HighlightButton(curButton);
        //             break;
        //         case "down":
        //             NormalButton(curButton);
        //             InputString = "";
        //             curIndex += 3;
        //             if (curIndex > 9)
        //                 curIndex -= 9;
        //             curButton = gridDict[curIndex.ToString()];
        //             HighlightButton(curButton);
        //             break;
        //         case "left":
        //             NormalButton(curButton);
        //             InputString = "";
        //             curIndex -= 1;
        //             if (curIndex == 0)
        //                 curIndex += 9;
        //             curButton = gridDict[curIndex.ToString()];
        //             HighlightButton(curButton);
        //             break;
        //         case "right":
        //             NormalButton(curButton);
        //             InputString = "";
        //             curIndex += 1;
        //             if (curIndex > 9)
        //                 curIndex -= 9;
        //             curButton = gridDict[curIndex.ToString()];
        //             HighlightButton(curButton);
        //             break;
        //         case "tap":                   
        //             NormalButton(curButton);
        //             InputString = "";
        //             PressButton(curButton);
        //             break;
        //         case "cclock":
        //             DemoApp.SetActive(false);
        //             InputString = "";
        //             break;
        //         case "clock":
        //             DemoApp.SetActive(true);
        //             InputString = "";
        //             break;
        //         default:
        //             break;
        //     }
        // }

        // if (mode == "music")
        // { 
        //     switch (InputString)
        //     {                
        //         case "up":
        //             InputString = "";
        //             break;
        //         case "down":
        //             InputString = "";
        //             break;
        //         case "left":
        //             InputString = "";
        //             textureIdx -= 1;
        //             if (textureIdx < 0)
        //                 textureIdx = 2;                                           
        //             img = (RawImage)musicDict["album"].GetComponent<RawImage>();
        //             img.texture = (Texture)TextureList[textureIdx];
        //             break;
        //         case "right":
        //             InputString = "";
        //             textureIdx += 1;
        //             if (textureIdx > 2)
        //                 textureIdx = 0;
                             
        //             img = (RawImage)musicDict["album"].GetComponent<RawImage>();
        //             img.texture = (Texture)TextureList[textureIdx];
        //             break;
        //         case "tap":
        //             flag_activate = !flag_activate;
        //             DemoApp.SetActive(flag_activate);
        //             InputString = "";
        //             break;
        //         case "cclock":
        //             StartCoroutine(DecreaseVolume());
        //             InputString = "";
        //             break;
        //         case "clock":
        //             StartCoroutine(IncreaseVolume());
        //             InputString = "";
        //             break;
        //         default:
        //             InputString = "";
        //             break;
        //     }
        // }

        // if (!scaling)
        // {            
        //     if (_upScale)
        //     {
        //         Debug.Log("upscale : " + slider.value);
        //         StartCoroutine(ScaleUpSpeaker());            
        //     }
        //     else
        //     {
        //         Debug.Log("downscale : " + slider.value);
        //         StartCoroutine(ScaleDownSpeaker());            
        //     }
        // }
        
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
                if (demo_step_idx == 0)
                {
                    PannelApp.SetActive(true);
                    pannelDict[curIndex.ToString()].Select();
                    demo_step_idx += 1;
                }
                else if (demo_step_idx == 1)
                {
                    pannelDict[curIndex.ToString()].onClick.Invoke();

                    var colors = pannelDict[curIndex.ToString()].colors;
                    pannelDict[curIndex.ToString()].targetGraphic.color = colors.pressedColor;
                    StartCoroutine(ResetColor(pannelDict[curIndex.ToString()], colors.disabledColor, 0.3f));
                    demo_step_idx += 1;
                }
                else if (demo_step_idx == 2)
                {
                    MusicApp.SetActive(false);
                    demo_step_idx += 1;
                }
                else if (demo_step_idx == 3)
                {
                    MusicApp.SetActive(true);
                    demo_step_idx += 1;
                }
                else if (demo_step_idx == 4)
                {
                    MusicApp.SetActive(false);
                    demo_step_idx += 1;
                }
                

                InputString = "";
                break;
            case "cclock":
                InputString = "";
                break;
            case "clock":
                InputString = "";
                break;
            default:
                break;
        }
        

        // Keypad 4 → Left
        if (Input.GetKeyUp(KeyCode.Keypad4) || Input.GetKeyUp(KeyCode.Alpha4))
        {
            InputString = "left";
            Debug.Log("입력: " + InputString);
        }

        // Keypad 6 → Right
        if (Input.GetKeyUp(KeyCode.Keypad6) || Input.GetKeyUp(KeyCode.Alpha6))
        {
            InputString = "right";
            Debug.Log("입력: " + InputString);
        }

        // Keypad 8 → Up
        if (Input.GetKeyUp(KeyCode.Keypad8) || Input.GetKeyUp(KeyCode.Alpha8))
        {
            InputString = "up";
            Debug.Log("입력: " + InputString);
        }

        // Keypad 2 → Down
        if (Input.GetKeyUp(KeyCode.Keypad2) || Input.GetKeyUp(KeyCode.Alpha2))
        {
            InputString = "down";
            Debug.Log("입력: " + InputString);
        }
        // Keypad 5 → Tap
        if (Input.GetKeyUp(KeyCode.Keypad5) || Input.GetKeyUp(KeyCode.Alpha5))
        {
            InputString = "tap";
            Debug.Log("입력: " + InputString);
        }

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

    
    public void event_MusicApp(GameObject TargetApp)
    {
        Debug.Log("Music App Activated");

        // mode = "music";
        PannelApp.SetActive(false);
        TargetApp.SetActive(true);

    }

    // private IEnumerator ScaleUpSpeaker()
    // {
    //     // float scale = ;

        
    //     float TargetScale = InitScale + maxScale*slider.value + 5.0f;
        
    //     float animTime = AnimationTimeSeconds - varAnimTime * slider.value;

    //     float _deltaTime = animTime/FramesCount;

    //     float _dx = (TargetScale - _currentScale)/FramesCount;

    //     while (true)
    //     {
    //         _currentScale += _dx;            
    //         speaker.transform.localScale = Vector3.one * _currentScale;        

    //         if (_currentScale > TargetScale)
    //         {
    //             _upScale = false;
    //             scaling = false;
    //             break;
    //         }
    //         yield return new WaitForSeconds(_deltaTime);    
    //     }
        
    // }
    
    // private IEnumerator ScaleDownSpeaker()
    // {
    //     // float scale = slider.value;
    //     float TargetScale = InitScale - maxScale*slider.value - 5.0f;

    //     float animTime = AnimationTimeSeconds - varAnimTime * slider.value;
    //     float _deltaTime = animTime/FramesCount;
    //     float _dx = (TargetScale - _currentScale)/FramesCount;

    //     while (true)
    //     {
    //         _currentScale += _dx;            
    //         speaker.transform.localScale = Vector3.one * _currentScale;        

    //         if (_currentScale < TargetScale)
    //         {
    //             _upScale = true;
    //             scaling = false;               
    //             break;
    //         }
    //         yield return new WaitForSeconds(_deltaTime);    
    //     }             
    // }
    
    
    // IEnumerator IncreaseVolume()
    // {
    //     for (int i =0; i < 40; i++)
    //     {
    //         if (slider.value < 1)
    //         {
    //             slider.value += 0.005f;
    //             yield return new WaitForSeconds(0.01f);
    //         }
    //     }    
    // }

    // IEnumerator DecreaseVolume()
    // {
    //     for (int i = 0; i < 40; i++)
    //     {
    //         if (slider.value > 0)
    //         {
    //             slider.value -= 0.005f;
    //             yield return new WaitForSeconds(0.01f);
    //         }
    //     }
    // }


    // void NormalButton(GameObject curButton)
    // {
    //     Animator animator = curButton.GetComponent<Animator>();
    //     animator.SetBool("Highlighted", false);
    //     animator.SetBool("Pressed", false);
    //     animator.SetBool("Normal", true);
    // }

    // void HighlightButton(GameObject  curButton)
    // {
    //     Animator animator = curButton.GetComponent<Animator>();
    //     animator.SetBool("Highlighted", true);
    // }

    // void PressButton(GameObject curButton)
    // {
    //     Animator animator = curButton.GetComponent<Animator>();
    //     animator.SetBool("Pressed", true);
    // }

    public void GetInputMessage(string inputMessage)
    {
        InputString = inputMessage;
    }
}
