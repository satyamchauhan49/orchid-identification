# Sneha Malik: model comparison and explainability

Validation results. Locked-test analysis remains a separate team stage.

## Models and protocol
EfficientNet-B0 and ConvNeXt-T share a 156-way species head and six feature heads (T, NF, In, LC, CF, CL). Missing labels are masked. Raw logits preserve the calibration interface.
Seed: 42. Resolution: 224 x 224. Optimizer: AdamW. Epochs: 15. Batch size: 32. Loss: ce. Sampler: plain. Pretrained: True.
Validation loss selects each run's checkpoint. The grid selects by mean feature macro-F1, breaking ties using validation loss. Final backbone selection still requires calibration and Agrim's ResNet comparison.

## Validation comparison
| Backbone | LR | Species macro-F1 | Feature macro-F1 | Top-1 | Val loss |
|---|---:|---:|---:|---:|---:|
| efficientnet_b0 | 0.001 | 0.5430 | 0.8516 | 0.7313 | 3.2032 |
| efficientnet_b0 | 0.0003 | 0.5380 | 0.8609 | 0.7581 | 2.7453 |
| efficientnet_b0 | 0.0001 | 0.5609 | 0.8632 | 0.7502 | 2.7985 |
| convnext_tiny | 0.001 | 0.0855 | 0.5321 | 0.3095 | 8.0611 |
| convnext_tiny | 0.0003 | 0.4875 | 0.8428 | 0.7243 | 3.1707 |
| convnext_tiny | 0.0001 | 0.6126 | 0.8971 | 0.7942 | 2.0541 |

Selected validation run: `convnext_tiny_lr0.0001`.

## R1 contribution: per-value feature metrics
TPR = TP/(TP+FN); TNR = TN/(TN+FP). Undefined rates are null. Macro-F1 averages labels present in predictions or ground truth.
| Run | Feature | Value | Support | TPR | TNR |
|---|---|---|---:|---:|---:|
| efficientnet_b0_lr0.001 | T | Nospot | 1007 | 0.9801 | 0.8158 |
| efficientnet_b0_lr0.001 | T | Spot | 266 | 0.8158 | 0.9801 |
| efficientnet_b0_lr0.001 | NF | AFew | 398 | 0.7588 | 0.9486 |
| efficientnet_b0_lr0.001 | NF | Many | 591 | 0.9357 | 0.8504 |
| efficientnet_b0_lr0.001 | NF | SinglePair | 284 | 0.9190 | 0.9899 |
| efficientnet_b0_lr0.001 | In | Panicle | 29 | 0.6897 | 0.9960 |
| efficientnet_b0_lr0.001 | In | Raceme | 708 | 0.9110 | 0.8867 |
| efficientnet_b0_lr0.001 | In | SingleOrPair | 285 | 0.9123 | 0.9899 |
| efficientnet_b0_lr0.001 | In | Spike | 251 | 0.8645 | 0.9491 |
| efficientnet_b0_lr0.001 | LC | Fringed | 111 | 0.8378 | 0.9897 |
| efficientnet_b0_lr0.001 | LC | Lobed | 516 | 0.9535 | 0.9062 |
| efficientnet_b0_lr0.001 | LC | Pouched | 296 | 0.8784 | 0.9857 |
| efficientnet_b0_lr0.001 | LC | Simple | 350 | 0.8543 | 0.9653 |
| efficientnet_b0_lr0.001 | CF | Brown | 3 | 0.3333 | 0.9984 |
| efficientnet_b0_lr0.001 | CF | Green | 220 | 0.8545 | 0.9516 |
| efficientnet_b0_lr0.001 | CF | GreenBrown | 298 | 0.8960 | 0.9795 |
| efficientnet_b0_lr0.001 | CF | GreenWhite | 10 | 0.6000 | 1.0000 |
| efficientnet_b0_lr0.001 | CF | GreenYellow | 38 | 0.7895 | 0.9935 |
| efficientnet_b0_lr0.001 | CF | Orange | 39 | 0.8718 | 0.9992 |
| efficientnet_b0_lr0.001 | CF | Pink | 156 | 0.9231 | 0.9955 |
| efficientnet_b0_lr0.001 | CF | PinkWhite | 6 | 0.3333 | 0.9992 |
| efficientnet_b0_lr0.001 | CF | Purple | 79 | 0.8228 | 0.9841 |
| efficientnet_b0_lr0.001 | CF | PurpleWhite | 6 | 1.0000 | 0.9976 |
| efficientnet_b0_lr0.001 | CF | Red | 35 | 0.8857 | 0.9927 |
| efficientnet_b0_lr0.001 | CF | White | 252 | 0.8413 | 0.9726 |
| efficientnet_b0_lr0.001 | CF | Yellow | 20 | 0.7000 | 0.9928 |
| efficientnet_b0_lr0.001 | CF | YellowBrown | 81 | 0.8272 | 0.9857 |
| efficientnet_b0_lr0.001 | CF | YellowPurple | 30 | 0.8667 | 0.9944 |
| efficientnet_b0_lr0.001 | CL | BluePurple | 122 | 0.8525 | 0.9705 |
| efficientnet_b0_lr0.001 | CL | Green | 175 | 0.8571 | 0.9408 |
| efficientnet_b0_lr0.001 | CL | Orange | 39 | 0.8974 | 1.0000 |
| efficientnet_b0_lr0.001 | CL | PinkRed | 246 | 0.9146 | 0.9737 |
| efficientnet_b0_lr0.001 | CL | PinkRedWhite | 92 | 0.7391 | 0.9958 |
| efficientnet_b0_lr0.001 | CL | PurpleYellow | 21 | 0.9524 | 1.0000 |
| efficientnet_b0_lr0.001 | CL | White | 442 | 0.8710 | 0.9471 |
| efficientnet_b0_lr0.001 | CL | WhiteYellow | 4 | 0.2500 | 0.9976 |
| efficientnet_b0_lr0.001 | CL | Yellow | 109 | 0.7431 | 0.9966 |
| efficientnet_b0_lr0.001 | CL | YellowBrown | 23 | 0.8261 | 0.9976 |
| efficientnet_b0_lr0.0003 | T | Nospot | 1007 | 0.9752 | 0.8571 |
| efficientnet_b0_lr0.0003 | T | Spot | 266 | 0.8571 | 0.9752 |
| efficientnet_b0_lr0.0003 | NF | AFew | 398 | 0.7839 | 0.9497 |
| efficientnet_b0_lr0.0003 | NF | Many | 591 | 0.9340 | 0.8680 |
| efficientnet_b0_lr0.0003 | NF | SinglePair | 284 | 0.9331 | 0.9899 |
| efficientnet_b0_lr0.0003 | In | Panicle | 29 | 0.7241 | 0.9960 |
| efficientnet_b0_lr0.0003 | In | Raceme | 708 | 0.9280 | 0.9062 |
| efficientnet_b0_lr0.0003 | In | SingleOrPair | 285 | 0.9333 | 0.9919 |
| efficientnet_b0_lr0.0003 | In | Spike | 251 | 0.8805 | 0.9589 |
| efficientnet_b0_lr0.0003 | LC | Fringed | 111 | 0.8198 | 0.9888 |
| efficientnet_b0_lr0.0003 | LC | Lobed | 516 | 0.9264 | 0.9287 |
| efficientnet_b0_lr0.0003 | LC | Pouched | 296 | 0.9426 | 0.9703 |
| efficientnet_b0_lr0.0003 | LC | Simple | 350 | 0.8314 | 0.9588 |
| efficientnet_b0_lr0.0003 | CF | Brown | 3 | 0.0000 | 1.0000 |
| efficientnet_b0_lr0.0003 | CF | Green | 220 | 0.8727 | 0.9601 |
| efficientnet_b0_lr0.0003 | CF | GreenBrown | 298 | 0.9228 | 0.9795 |
| efficientnet_b0_lr0.0003 | CF | GreenWhite | 10 | 0.8000 | 0.9984 |
| efficientnet_b0_lr0.0003 | CF | GreenYellow | 38 | 0.7895 | 0.9943 |
| efficientnet_b0_lr0.0003 | CF | Orange | 39 | 0.8974 | 0.9992 |
| efficientnet_b0_lr0.0003 | CF | Pink | 156 | 0.9615 | 0.9902 |
| efficientnet_b0_lr0.0003 | CF | PinkWhite | 6 | 0.3333 | 1.0000 |
| efficientnet_b0_lr0.0003 | CF | Purple | 79 | 0.8101 | 0.9941 |
| efficientnet_b0_lr0.0003 | CF | PurpleWhite | 6 | 0.6667 | 1.0000 |
| efficientnet_b0_lr0.0003 | CF | Red | 35 | 0.9429 | 0.9952 |
| efficientnet_b0_lr0.0003 | CF | White | 252 | 0.8929 | 0.9657 |
| efficientnet_b0_lr0.0003 | CF | Yellow | 20 | 0.7500 | 0.9936 |
| efficientnet_b0_lr0.0003 | CF | YellowBrown | 81 | 0.9012 | 0.9966 |
| efficientnet_b0_lr0.0003 | CF | YellowPurple | 30 | 0.7667 | 0.9992 |
| efficientnet_b0_lr0.0003 | CL | BluePurple | 122 | 0.8115 | 0.9861 |
| efficientnet_b0_lr0.0003 | CL | Green | 175 | 0.8057 | 0.9563 |
| efficientnet_b0_lr0.0003 | CL | Orange | 39 | 0.8974 | 0.9992 |
| efficientnet_b0_lr0.0003 | CL | PinkRed | 246 | 0.9024 | 0.9815 |
| efficientnet_b0_lr0.0003 | CL | PinkRedWhite | 92 | 0.8261 | 0.9924 |
| efficientnet_b0_lr0.0003 | CL | PurpleYellow | 21 | 0.9524 | 1.0000 |
| efficientnet_b0_lr0.0003 | CL | White | 442 | 0.9140 | 0.9266 |
| efficientnet_b0_lr0.0003 | CL | WhiteYellow | 4 | 0.2500 | 1.0000 |
| efficientnet_b0_lr0.0003 | CL | Yellow | 109 | 0.7890 | 0.9871 |
| efficientnet_b0_lr0.0003 | CL | YellowBrown | 23 | 0.6957 | 0.9968 |
| efficientnet_b0_lr0.0001 | T | Nospot | 1007 | 0.9662 | 0.8195 |
| efficientnet_b0_lr0.0001 | T | Spot | 266 | 0.8195 | 0.9662 |
| efficientnet_b0_lr0.0001 | NF | AFew | 398 | 0.8166 | 0.9326 |
| efficientnet_b0_lr0.0001 | NF | Many | 591 | 0.9036 | 0.8915 |
| efficientnet_b0_lr0.0001 | NF | SinglePair | 284 | 0.9577 | 0.9909 |
| efficientnet_b0_lr0.0001 | In | Panicle | 29 | 0.7586 | 0.9944 |
| efficientnet_b0_lr0.0001 | In | Raceme | 708 | 0.9364 | 0.9027 |
| efficientnet_b0_lr0.0001 | In | SingleOrPair | 285 | 0.9474 | 0.9899 |
| efficientnet_b0_lr0.0001 | In | Spike | 251 | 0.8526 | 0.9687 |
| efficientnet_b0_lr0.0001 | LC | Fringed | 111 | 0.8829 | 0.9862 |
| efficientnet_b0_lr0.0001 | LC | Lobed | 516 | 0.9147 | 0.9617 |
| efficientnet_b0_lr0.0001 | LC | Pouched | 296 | 0.9155 | 0.9806 |
| efficientnet_b0_lr0.0001 | LC | Simple | 350 | 0.9371 | 0.9567 |
| efficientnet_b0_lr0.0001 | CF | Brown | 3 | 0.0000 | 1.0000 |
| efficientnet_b0_lr0.0001 | CF | Green | 220 | 0.8091 | 0.9687 |
| efficientnet_b0_lr0.0001 | CF | GreenBrown | 298 | 0.9195 | 0.9733 |
| efficientnet_b0_lr0.0001 | CF | GreenWhite | 10 | 0.7000 | 0.9984 |
| efficientnet_b0_lr0.0001 | CF | GreenYellow | 38 | 0.7895 | 0.9935 |
| efficientnet_b0_lr0.0001 | CF | Orange | 39 | 0.9231 | 0.9992 |
| efficientnet_b0_lr0.0001 | CF | Pink | 156 | 0.9679 | 0.9937 |
| efficientnet_b0_lr0.0001 | CF | PinkWhite | 6 | 0.5000 | 0.9984 |
| efficientnet_b0_lr0.0001 | CF | Purple | 79 | 0.8101 | 0.9941 |
| efficientnet_b0_lr0.0001 | CF | PurpleWhite | 6 | 1.0000 | 1.0000 |
| efficientnet_b0_lr0.0001 | CF | Red | 35 | 0.8857 | 0.9968 |
| efficientnet_b0_lr0.0001 | CF | White | 252 | 0.9206 | 0.9598 |
| efficientnet_b0_lr0.0001 | CF | Yellow | 20 | 0.7500 | 0.9960 |
| efficientnet_b0_lr0.0001 | CF | YellowBrown | 81 | 0.8519 | 0.9916 |
| efficientnet_b0_lr0.0001 | CF | YellowPurple | 30 | 0.8000 | 0.9944 |
| efficientnet_b0_lr0.0001 | CL | BluePurple | 122 | 0.7869 | 0.9896 |
| efficientnet_b0_lr0.0001 | CL | Green | 175 | 0.8171 | 0.9672 |
| efficientnet_b0_lr0.0001 | CL | Orange | 39 | 0.9231 | 0.9984 |
| efficientnet_b0_lr0.0001 | CL | PinkRed | 246 | 0.8984 | 0.9854 |
| efficientnet_b0_lr0.0001 | CL | PinkRedWhite | 92 | 0.8478 | 0.9881 |
| efficientnet_b0_lr0.0001 | CL | PurpleYellow | 21 | 0.9524 | 1.0000 |
| efficientnet_b0_lr0.0001 | CL | White | 442 | 0.9321 | 0.9110 |
| efficientnet_b0_lr0.0001 | CL | WhiteYellow | 4 | 0.2500 | 1.0000 |
| efficientnet_b0_lr0.0001 | CL | Yellow | 109 | 0.7706 | 0.9914 |
| efficientnet_b0_lr0.0001 | CL | YellowBrown | 23 | 0.6522 | 0.9968 |
| convnext_tiny_lr0.001 | T | Nospot | 1007 | 0.9345 | 0.4474 |
| convnext_tiny_lr0.001 | T | Spot | 266 | 0.4474 | 0.9345 |
| convnext_tiny_lr0.001 | NF | AFew | 398 | 0.2462 | 0.9291 |
| convnext_tiny_lr0.001 | NF | Many | 591 | 0.8968 | 0.5205 |
| convnext_tiny_lr0.001 | NF | SinglePair | 284 | 0.6831 | 0.9373 |
| convnext_tiny_lr0.001 | In | Panicle | 29 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | In | Raceme | 708 | 0.8672 | 0.5575 |
| convnext_tiny_lr0.001 | In | SingleOrPair | 285 | 0.6561 | 0.9585 |
| convnext_tiny_lr0.001 | In | Spike | 251 | 0.4422 | 0.9315 |
| convnext_tiny_lr0.001 | LC | Fringed | 111 | 0.2973 | 0.9664 |
| convnext_tiny_lr0.001 | LC | Lobed | 516 | 0.8314 | 0.6341 |
| convnext_tiny_lr0.001 | LC | Pouched | 296 | 0.4865 | 0.9324 |
| convnext_tiny_lr0.001 | LC | Simple | 350 | 0.4714 | 0.8700 |
| convnext_tiny_lr0.001 | CF | Brown | 3 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | CF | Green | 220 | 0.6227 | 0.8898 |
| convnext_tiny_lr0.001 | CF | GreenBrown | 298 | 0.6611 | 0.8595 |
| convnext_tiny_lr0.001 | CF | GreenWhite | 10 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | CF | GreenYellow | 38 | 0.1316 | 0.9968 |
| convnext_tiny_lr0.001 | CF | Orange | 39 | 0.7436 | 0.9895 |
| convnext_tiny_lr0.001 | CF | Pink | 156 | 0.8269 | 0.9624 |
| convnext_tiny_lr0.001 | CF | PinkWhite | 6 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | CF | Purple | 79 | 0.4051 | 0.9899 |
| convnext_tiny_lr0.001 | CF | PurpleWhite | 6 | 0.0000 | 0.9992 |
| convnext_tiny_lr0.001 | CF | Red | 35 | 0.4286 | 0.9822 |
| convnext_tiny_lr0.001 | CF | White | 252 | 0.7222 | 0.9021 |
| convnext_tiny_lr0.001 | CF | Yellow | 20 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | CF | YellowBrown | 81 | 0.4691 | 0.9815 |
| convnext_tiny_lr0.001 | CF | YellowPurple | 30 | 0.5000 | 0.9799 |
| convnext_tiny_lr0.001 | CL | BluePurple | 122 | 0.3607 | 0.9748 |
| convnext_tiny_lr0.001 | CL | Green | 175 | 0.5371 | 0.8953 |
| convnext_tiny_lr0.001 | CL | Orange | 39 | 0.7179 | 0.9895 |
| convnext_tiny_lr0.001 | CL | PinkRed | 246 | 0.6829 | 0.9124 |
| convnext_tiny_lr0.001 | CL | PinkRedWhite | 92 | 0.5000 | 0.9585 |
| convnext_tiny_lr0.001 | CL | PurpleYellow | 21 | 0.2381 | 0.9984 |
| convnext_tiny_lr0.001 | CL | White | 442 | 0.7014 | 0.7545 |
| convnext_tiny_lr0.001 | CL | WhiteYellow | 4 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.001 | CL | Yellow | 109 | 0.5046 | 0.9897 |
| convnext_tiny_lr0.001 | CL | YellowBrown | 23 | 0.1739 | 0.9960 |
| convnext_tiny_lr0.0003 | T | Nospot | 1007 | 0.9454 | 0.8722 |
| convnext_tiny_lr0.0003 | T | Spot | 266 | 0.8722 | 0.9454 |
| convnext_tiny_lr0.0003 | NF | AFew | 398 | 0.7864 | 0.9223 |
| convnext_tiny_lr0.0003 | NF | Many | 591 | 0.8900 | 0.8768 |
| convnext_tiny_lr0.0003 | NF | SinglePair | 284 | 0.9366 | 0.9838 |
| convnext_tiny_lr0.0003 | In | Panicle | 29 | 0.7586 | 0.9944 |
| convnext_tiny_lr0.0003 | In | Raceme | 708 | 0.9138 | 0.9097 |
| convnext_tiny_lr0.0003 | In | SingleOrPair | 285 | 0.9439 | 0.9808 |
| convnext_tiny_lr0.0003 | In | Spike | 251 | 0.8526 | 0.9569 |
| convnext_tiny_lr0.0003 | LC | Fringed | 111 | 0.8739 | 0.9811 |
| convnext_tiny_lr0.0003 | LC | Lobed | 516 | 0.8779 | 0.9564 |
| convnext_tiny_lr0.0003 | LC | Pouched | 296 | 0.8615 | 0.9949 |
| convnext_tiny_lr0.0003 | LC | Simple | 350 | 0.9543 | 0.9198 |
| convnext_tiny_lr0.0003 | CF | Brown | 3 | 0.0000 | 0.9992 |
| convnext_tiny_lr0.0003 | CF | Green | 220 | 0.8545 | 0.9544 |
| convnext_tiny_lr0.0003 | CF | GreenBrown | 298 | 0.9128 | 0.9785 |
| convnext_tiny_lr0.0003 | CF | GreenWhite | 10 | 0.8000 | 0.9976 |
| convnext_tiny_lr0.0003 | CF | GreenYellow | 38 | 0.6579 | 0.9968 |
| convnext_tiny_lr0.0003 | CF | Orange | 39 | 0.9231 | 0.9984 |
| convnext_tiny_lr0.0003 | CF | Pink | 156 | 0.8782 | 0.9973 |
| convnext_tiny_lr0.0003 | CF | PinkWhite | 6 | 0.5000 | 0.9984 |
| convnext_tiny_lr0.0003 | CF | Purple | 79 | 0.8734 | 0.9782 |
| convnext_tiny_lr0.0003 | CF | PurpleWhite | 6 | 0.6667 | 1.0000 |
| convnext_tiny_lr0.0003 | CF | Red | 35 | 0.8571 | 1.0000 |
| convnext_tiny_lr0.0003 | CF | White | 252 | 0.8452 | 0.9794 |
| convnext_tiny_lr0.0003 | CF | Yellow | 20 | 0.7500 | 0.9872 |
| convnext_tiny_lr0.0003 | CF | YellowBrown | 81 | 0.9136 | 0.9832 |
| convnext_tiny_lr0.0003 | CF | YellowPurple | 30 | 0.9333 | 0.9968 |
| convnext_tiny_lr0.0003 | CL | BluePurple | 122 | 0.7951 | 0.9818 |
| convnext_tiny_lr0.0003 | CL | Green | 175 | 0.8514 | 0.9536 |
| convnext_tiny_lr0.0003 | CL | Orange | 39 | 0.9231 | 0.9992 |
| convnext_tiny_lr0.0003 | CL | PinkRed | 246 | 0.8984 | 0.9805 |
| convnext_tiny_lr0.0003 | CL | PinkRedWhite | 92 | 0.8478 | 0.9949 |
| convnext_tiny_lr0.0003 | CL | PurpleYellow | 21 | 0.8095 | 1.0000 |
| convnext_tiny_lr0.0003 | CL | White | 442 | 0.8937 | 0.9446 |
| convnext_tiny_lr0.0003 | CL | WhiteYellow | 4 | 0.0000 | 1.0000 |
| convnext_tiny_lr0.0003 | CL | Yellow | 109 | 0.8257 | 0.9820 |
| convnext_tiny_lr0.0003 | CL | YellowBrown | 23 | 0.8261 | 0.9960 |
| convnext_tiny_lr0.0001 | T | Nospot | 1007 | 0.9811 | 0.9023 |
| convnext_tiny_lr0.0001 | T | Spot | 266 | 0.9023 | 0.9811 |
| convnext_tiny_lr0.0001 | NF | AFew | 398 | 0.8844 | 0.9326 |
| convnext_tiny_lr0.0001 | NF | Many | 591 | 0.9103 | 0.9311 |
| convnext_tiny_lr0.0001 | NF | SinglePair | 284 | 0.9648 | 0.9970 |
| convnext_tiny_lr0.0001 | In | Panicle | 29 | 0.7241 | 0.9992 |
| convnext_tiny_lr0.0001 | In | Raceme | 708 | 0.9732 | 0.9080 |
| convnext_tiny_lr0.0001 | In | SingleOrPair | 285 | 0.9649 | 0.9970 |
| convnext_tiny_lr0.0001 | In | Spike | 251 | 0.8566 | 0.9834 |
| convnext_tiny_lr0.0001 | LC | Fringed | 111 | 0.8559 | 0.9966 |
| convnext_tiny_lr0.0001 | LC | Lobed | 516 | 0.9554 | 0.9538 |
| convnext_tiny_lr0.0001 | LC | Pouched | 296 | 0.9189 | 0.9949 |
| convnext_tiny_lr0.0001 | LC | Simple | 350 | 0.9371 | 0.9556 |
| convnext_tiny_lr0.0001 | CF | Brown | 3 | 0.0000 | 0.9992 |
| convnext_tiny_lr0.0001 | CF | Green | 220 | 0.9182 | 0.9668 |
| convnext_tiny_lr0.0001 | CF | GreenBrown | 298 | 0.9530 | 0.9805 |
| convnext_tiny_lr0.0001 | CF | GreenWhite | 10 | 0.9000 | 0.9984 |
| convnext_tiny_lr0.0001 | CF | GreenYellow | 38 | 0.7368 | 0.9976 |
| convnext_tiny_lr0.0001 | CF | Orange | 39 | 0.8974 | 1.0000 |
| convnext_tiny_lr0.0001 | CF | Pink | 156 | 1.0000 | 0.9928 |
| convnext_tiny_lr0.0001 | CF | PinkWhite | 6 | 0.6667 | 0.9976 |
| convnext_tiny_lr0.0001 | CF | Purple | 79 | 0.8861 | 0.9958 |
| convnext_tiny_lr0.0001 | CF | PurpleWhite | 6 | 1.0000 | 0.9992 |
| convnext_tiny_lr0.0001 | CF | Red | 35 | 0.9143 | 0.9984 |
| convnext_tiny_lr0.0001 | CF | White | 252 | 0.8929 | 0.9863 |
| convnext_tiny_lr0.0001 | CF | Yellow | 20 | 0.7500 | 0.9976 |
| convnext_tiny_lr0.0001 | CF | YellowBrown | 81 | 0.8889 | 0.9950 |
| convnext_tiny_lr0.0001 | CF | YellowPurple | 30 | 0.9000 | 0.9952 |
| convnext_tiny_lr0.0001 | CL | BluePurple | 122 | 0.8770 | 0.9948 |
| convnext_tiny_lr0.0001 | CL | Green | 175 | 0.8800 | 0.9690 |
| convnext_tiny_lr0.0001 | CL | Orange | 39 | 0.9231 | 1.0000 |
| convnext_tiny_lr0.0001 | CL | PinkRed | 246 | 0.9390 | 0.9864 |
| convnext_tiny_lr0.0001 | CL | PinkRedWhite | 92 | 0.8696 | 0.9932 |
| convnext_tiny_lr0.0001 | CL | PurpleYellow | 21 | 0.9524 | 1.0000 |
| convnext_tiny_lr0.0001 | CL | White | 442 | 0.9548 | 0.9362 |
| convnext_tiny_lr0.0001 | CL | WhiteYellow | 4 | 0.2500 | 1.0000 |
| convnext_tiny_lr0.0001 | CL | Yellow | 109 | 0.7982 | 0.9983 |
| convnext_tiny_lr0.0001 | CL | YellowBrown | 23 | 0.7826 | 1.0000 |

## Explainability method
Grad-CAM uses gradients from the predicted species/CF/CL logit at the selected final spatial layer. Heatmaps and the resized input use the same 224-pixel coordinates. Real data uses checkpoint-stored train-only mean/std. Softmax scores are uncalibrated.
Taxonomic records compare observed features to expected feature distributions. Contrastive records rank feature differences by total variation and include candidate references. Features below 0.4 confidence are marked unobserved. Decision records retain the externally supplied stage, confidence and reason.

## Example records
| Image | Stage | DNN candidate | BN candidate |
|---|---|---|---|
| 11.jpg | trusted | 58 | 58 |
| 19.jpg | deferred | 1 | 58 |
| 22.jpg | deferred | 58 | 58 |
| 4.jpg | trusted | 1 | 1 |
| 5.jpg | deferred | 1 | 1 |
| 28.jpg | deferred | 23 | 53 |
| 33.jpg | deferred | 23 | 23 |
| 37.jpg | deferred | 23 | 53 |
| 116.jpg | trusted | 3 | 3 |
| 1398.jpg | auto_resolved | 25 | 145 |

## Limitations and remaining team work
Real validation under Satyam's disjoint-calibration policy: DNN/fusion accuracy 79.42%, accepted accuracy 94.46%, deferral 30.48%. Weight 1.0 means fusion equals DNN; no BN gain is demonstrated. There are 877 trusted, 8 auto_resolved and 388 deferred decisions overall. Accepted error exceeds the 1.5% target. Stage names alone do not prove BN corrections.

Ten actual validation cases are illustrated with matching model Grad-CAM. Consistent labellum localization is not established. Taxonomy is empirical TRAIN evidence. These are not final test results. Team freeze, locked evaluation and publication remain pending. Earlier 79.58%/96.86%/39.98% figures belong to a separate policy experiment.

## Input provenance
{
  "manifest": "bf8d843fc84203cd7aea74904aa153d7d5c71a0e1dd9d2861554c6e9d6e5a7a5",
  "label_maps": "c523cf61d55f30f452653056e95e0af92f8dbca9714749e0afae65dc633708cc",
  "normalization": "f80befd564d608073e700b0370264b191f361968685acaea8f42f78509740e0c"
}