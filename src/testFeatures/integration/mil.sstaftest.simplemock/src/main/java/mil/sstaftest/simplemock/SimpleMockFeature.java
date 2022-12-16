/*
 * Copyright (c) 2022
 * United States Government as represented by the U.S. Army DEVCOM Analysis Center.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mil.sstaftest.simplemock;

import mil.sstaf.core.features.BaseFeature;

public class SimpleMockFeature extends BaseFeature {
    /**
     * Constructor for subclasses.
     * <p>
     * Note that concrete implementations must have a no-args constructor to be loadable as services
     */
    protected SimpleMockFeature() {
        super("Simple Mock Feature",
                1, 0, 0,
                false, "This is nothing");
    }
}

